'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
from collections import OrderedDict
from sklearn.model_selection import KFold

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(comment='test_run2')


class Method_MLP(method, nn.Module):
    data = {
        'sLossFunction': 'CrossEntropy',
        'sOptimizer': 'ADAM',
        'sInputDim': 784,
        'sOutputDim': 10,
        'sLearningRate': 1e-3,
        'sMomentum': 0.9,
        'sMaxEpoch': 2  # ! CHANGE LATER
    }
    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output

    def __init__(self, mName='', mDescription='', sData=None):

        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        if sData:
            for k, v in sData.items():
                self.data[k] = v

        self.inputLayer = OrderedDict([
            ('fc_layer_1', nn.Linear(self.data['sInputDim'], 392)),
            ('activation_func_1', nn.ReLU()),
        ])
        self.hiddenLayers = OrderedDict([
            ('fc_layer_2', nn.Linear(392, 196)),
            ('activation_func_2', nn.ReLU()),
            ('fc_layer_3', nn.Linear(196, 98)),
            ('activation_func_3', nn.ReLU()),
            ('fc_layer_4', nn.Linear(98, 49)),
            ('activation_func_4', nn.ReLU()),
        ])   # add more layers later

        self.outputLayer = OrderedDict([
            ('fc_layer_5', nn.Linear(49, self.data['sOutputDim']))
        ])
        # do not use softmax if we have nn.CrossEntropyLoss base on the PyTorch documents
        if self.data['sLossFunction'] != 'CrossEntropy':
            self.outputLayer['activation_func_5'] = nn.Softmax(dim=1)
        else:
            self.outputLayer['activation_func_5'] = nn.ReLU()

        # Compile all layers
        self.layers = nn.ModuleDict(self.compileLayers())
        self.cuda()

        if self.data['sLossFunction'] == 'MSE':
            self.lossFunction = nn.MSELoss()
        else:
            self.lossFunction = nn.CrossEntropyLoss()

        if self.data['sOptimizer'] == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(),
                                             lr=self.data['sLearningRate'],
                                             momentum=self.data['sMomentum'])
        else:
            self.optimizer = torch.optim.Adam(self.parameters(),
                                              lr=self.data['sLearningRate'])
        self.lossList = []  # for plotting loss

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def compileLayers(self) -> OrderedDict:
        res = OrderedDict()
        res.update(self.inputLayer)
        if self.hiddenLayers:
            res.update(self.hiddenLayers)
        res.update(self.outputLayer)
        return res

    def forward(self, x):
        '''Forward propagation'''
        out = x
        for _, func in self.layers.items():
            out = func(out)

        return out

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def trainModel(self, X, y):
        # Turn on train mode for all layers and prepare data

        self.training = True
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = self.optimizer
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = self.lossFunction
        # for training accuracy investigation purpose

        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        # you can do an early stop if self.max_epoch is too much...
        kfold = KFold(
            n_splits=self.data['kfold'], shuffle=True, random_state=self.data['randSeed'])

        for epoch in range(self.data['sMaxEpoch']):
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            fold_pred = self.forward(torch.FloatTensor(
                np.array(X)).to(torch.device('cuda:0')))

            # convert y to torch.tensor as well
            fold_true = torch.LongTensor(
                np.array(y)).to(torch.device('cuda:0'))

            # calculate the training loss
            train_loss = loss_function(fold_pred, fold_true)
            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            if epoch % 100 == 0:
                # The y_pred.max(1)[1] return the indices of max value on each row of a tensor (the y_pred is a tensor)
                accuracy_evaluator.data = {
                    'true_y': fold_true.cpu(), 'pred_y': fold_pred.cpu().max(dim=1)[1]}
                # accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(dim=1)}
                acc = accuracy_evaluator.evaluate()
                loss = train_loss.item()
                print('Epoch:', epoch, 'Accuracy:',
                      acc, 'Loss:', loss)

            #Record data for ploting
            self.lossList.append(loss)

            # Record data for tensorboard
            writer.add_scalar('Training Loss', train_loss, epoch)
            writer.add_scalar('Accuracy', acc, epoch)

            # Check learning progress
            for name, weight in self.named_parameters():
                writer.add_histogram(name, weight, epoch)
                writer.add_histogram(f'{name}.grad', weight.grad, epoch)

    def test(self, X):
        # Set to test mode
        self.training = False
        # do the testing, and result the result
        with torch.no_grad():
            y_pred = self.forward(torch.FloatTensor(
                np.array(X)).to(torch.device('cuda:0')))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]


    def run(self):
        # Visualize the architecture
        test_data = torch.FloatTensor(np.array(self.data['train']['X'])).cuda()
        writer.add_graph(self, test_data)

        print('method running...')
        print('--start training...')
        self.trainModel(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X']).cpu()
        return {'pred_y': pred_y, 'true_y': self.data['test']['y'],
                'loss': self.lossList}
