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
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter


class Method_Classification(method, nn.Module):
    data = {
        'sLossFunction': 'CrossEntropy',
        'sOptimizer': 'ADAM',
        'sInputSize': 768, # BERT
        'sDropout': 0.5,
        'sOutputDim': 2,    # Binary classification
        'sLearningRate': 1e-3,
        'sMomentum': 0.9,
        'sMaxEpoch': 1,  # ! CHANGE LATER
        'sBatchSize': 5000,  # Lower than 4000 is required
        'sRandSeed': 47
    }

    def __init__(self, mName='', mDescription='', sData=None):

        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        if sData:
            for k, v in sData.items():
                self.data[k] = v

        self.model_res_name = 'rnn_models'

        for k, v in self.data.items():
            if k != 'train' and k != 'test':
                self.model_res_name += f'_{k}:{v}'
        self.writer = SummaryWriter(
            comment=self.model_res_name)
        
        self.inputLayer = OrderedDict([
            ('rnn_layer_1', nn.RNN(input_size=self.data['sInputSize'], hidden_size=512, batch_first=True))
        ])
        self.outputLayer = OrderedDict([
            ('flatten_layer_2', nn.Flatten()),
            ('linear_layer_2', nn.Linear(512, self.data['sOutputDim'])) # ! Potentially wrong size, change the input later
        ])

        # do not use softmax if we have nn.CrossEntropyLoss base on the PyTorch documents
        # ? https://stackoverflow.com/questions/55675345/should-i-use-softmax-as-output-when-using-cross-entropy-loss-in-pytorch
        if self.data['sLossFunction'] != 'CrossEntropy':
            self.outputLayer['output'] = nn.Softmax(dim=1)
        else:
            self.outputLayer['output'] = nn.ReLU()

        # Compile all layers
        self.layers = nn.ModuleDict(self.compileLayers())

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

        self.cuda()

    def compileLayers(self) -> OrderedDict:
        res = OrderedDict()
        res.update(self.inputLayer)
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
        optimizer = self.optimizer
        loss_function = self.lossFunction

        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')


        for epoch in range(self.data['sMaxEpoch']):
            # TODO: use Jai stackoverflow to put in batch loop --------------
            permutation = torch.randperm(len(X))    # random order of batches
            for i in range(0, len(X), self.data['sBatchSize']):
                indices = permutation[i:i+self.data['sBatchSize']]
                batchX, batchY = [X[i] for i in indices], [y[i] for i in indices]   # batches

                # ! Begin forward here
                fold_pred = self.forward(torch.FloatTensor(np.array(batchX)).cuda())

                fold_true = torch.LongTensor(
                    np.array(batchY)).cuda()

                # calculate the training loss
                train_loss = loss_function(fold_pred, fold_true)
                optimizer.zero_grad()
                train_loss.backward()
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

            # Record data for ploting
            self.lossList.append(loss)

            # Record data for tensorboard
            self.writer.add_scalar('Training Loss', train_loss, epoch)
            self.writer.add_scalar('Accuracy', acc, epoch)

            # Check learning progress
            for name, weight in self.named_parameters():
                self.writer.add_histogram(name, weight, epoch)
                self.writer.add_histogram(f'{name}.grad', weight.grad, epoch)

    def test(self, X):
        # Set to test mode
        y_predTotal = []
        self.training = False
        with torch.no_grad():
            for i in range(0, len(X), self.data['sBatchSize']):
                batchX = X[i:i+self.data['sBatchSize']]
                inTensor = torch.FloatTensor(np.array(batchX)).permute(0, 3, 1, 2)
                y_predTotal.extend(self.forward(inTensor.cuda()).cpu().max(dim=1)[1])

        return y_predTotal

    def run(self):
        #! Visualize the architecture
        tempX = self.data['train']['X'][0]
        boardGraphInput = torch.FloatTensor(np.array(tempX))
        print(f'Shape of input: {boardGraphInput.shape}')

        self.writer.add_graph(self, boardGraphInput.cuda())

        #! Actual run
        print('method running...')
        print('--network status--')
        netStats = summary(self, (1,self.data['sInChannels'],min(self.data['sInputDim']),min(self.data['sInputDim'])))
        print('--start training...')
        self.trainModel(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])

        # print(f'{pred_y} and the length {len(pred_y)} also length of each')
        # ALso for tensor
        return {'pred_y': pred_y, 'true_y': self.data['test']['y'],
                'loss': self.lossList}
