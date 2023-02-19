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


from torch.utils.tensorboard import SummaryWriter


class Method_MLP(method, nn.Module):
    data = {
        # 'sInitMethod': 'kaiming',
        'sLossFunction': 'CrossEntropy',
        'sOptimizer': 'ADAM',
        'sInputDim': (28, 28),
        'sInChannels': 1,   # Color channel (number of filter)
        'sDropout': 0.5,
        'sOutputDim': 10,
        'sLearningRate': 1e-4,
        'sMomentum': 0.9,
        'sMaxEpoch': 500,  # ! CHANGE LATER
        'sBatchSize': 1000
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
        # to keep the initialization stable
        # torch.manual_seed(self.data['sRandSeed'])

        self.model_res_name = 'nn_models'
        for k, v in self.data.items():
            if k != 'train' and k != 'test':
                self.model_res_name += f'_{k}:{v}'
        self.writer = SummaryWriter(
            comment=self.model_res_name)

        self.inputLayer = OrderedDict([  # ! Come to this CONV2d later
            ('conv_layer_1', nn.Conv2d(in_channels=self.data['sInChannels'],
                                       out_channels=64, kernel_size=11,
                                       stride=4, padding=2)),
            ('activation_func_1', nn.ReLU()),
            ('maxpool_layer_1', nn.MaxPool2d(kernel_size=3, stride=2))
        ])
        self.features = OrderedDict([
            ('conv_layer_2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
            ('activation_func_2', nn.ReLU()),
            ('maxpool_layer_2', nn.MaxPool2d(kernel_size=3, stride=2)),

            ('conv_layer_3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
            ('activation_func_3', nn.ReLU()),

            ('conv_layer_4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
            ('activation_func_4', nn.ReLU()),

            ('conv_layer_5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            ('activation_func_5', nn.ReLU()),
            ('maxpool_layer_5', nn.MaxPool2d(kernel_size=3, stride=2)),

            ('avgpool_layer_6', nn.AdaptiveAvgPool2d((6, 6)))
        ])   # add more layers later
        self.classifier = OrderedDict([
            ('dropout_layer_7', nn.Dropout(p=self.data['sDropout'])),
            ('speical_flat_layer_7', nn.Flatten()), # ! MAYYYYYBE
            ('linear_layer_7', nn.Linear((256 * 6 * 6), 4096)),
            ('activation_func_7', nn.ReLU()),

            ('dropout_layer_8', nn.Dropout(p=self.data['sDropout'])),
            ('linear_layer_8', nn.Linear(4096, 4096)),
            ('activation_func_8', nn.ReLU())
        ])
        self.outputLayer = OrderedDict([
            ('linear_layer_9', nn.Linear(4096, self.data['sOutputDim']))
        ])

        # do not use softmax if we have nn.CrossEntropyLoss base on the PyTorch documents
        # ? https://stackoverflow.com/questions/55675345/should-i-use-softmax-as-output-when-using-cross-entropy-loss-in-pytorch
        if self.data['sLossFunction'] != 'CrossEntropy':
            self.outputLayer['output'] = nn.Softmax(dim=1)
        else:
            self.outputLayer['output'] = nn.ReLU()

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

    def compileLayers(self) -> OrderedDict:
        res = OrderedDict()
        res.update(self.inputLayer)
        res.update(self.features)
        res.update(self.classifier)
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


        for epoch in range(self.data['sMaxEpoch']):
            # TODO: use Jai stackoverflow to put in batch loop --------------
            permutation = torch.randperm(len(X))    # random order of batches

            for i in range(0, len(X), self.data['sBatchSize']):
                indices = permutation[i:i+self.data['sBatchSize']]
                batchX, batchY = [X[i] for i in indices], [y[i] for i in indices]   # batches 
                
                if self.data['sDataName'] == 'MNIST':
                    inTensor = torch.FloatTensor(
                        np.array(batchX)).reshape(shape=(len(batchX), self.data['sInChannels'], self.data['sInputDim'][0], self.data['sInputDim'][1]))
                else:
                    #! Have to switch the color channel position in tensor for the other dataset
                    inTensor = torch.FloatTensor(np.array(batchX)).permute(
                        2, 0, 1).reshape(shape=(len(batchX), self.data['sInChannels'], self.data['sInputDim'][0], self.data['sInputDim'][1]))

                # * The input dimension of AlexNet is 224*224, thus we need to transform our input data
                # ? https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0263-7#Sec8
                    # ? The paper above suggest padding input image with zeroes a viable way to increase our image size
                    # ? Sadly, the easy way is not yet ready for pytorch: https://github.com/pytorch/vision/issues/6236

                # ? https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html#torch.nn.functional.pad
                heightPad = (224 - inTensor.shape[2]) // 2
                widthPad = (224 - inTensor.shape[3]) // 2
                padding = (heightPad, heightPad, widthPad, widthPad)
                # increase the image size by padding 0
                inTensor = nn.functional.pad(inTensor, padding, 'constant', 0)

                # ! Begin forward here
                fold_pred = self.forward(inTensor.cuda())

                # convert y to torch.tensor as well
                fold_true = torch.LongTensor(
                    np.array(batchY)).cuda()

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

            if epoch % 10 == 0:
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
        # TODO: mini batch test also
        y_predTotal = []
        self.training = False
        with torch.no_grad():
            for i in range(0, len(X), self.data['sBatchSize']):
                # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
                batchX = X[i:i+self.data['sBatchSize']]
                if self.data['sDataName'] == 'MNIST':
                    inTensor = torch.FloatTensor(
                        np.array(batchX)).reshape(shape=(len(batchX), self.data['sInChannels'], self.data['sInputDim'][0], self.data['sInputDim'][1]))
                else:
                    #! Have to switch the color channel position in tensor for the other dataset
                    inTensor = torch.FloatTensor(np.array(batchX)).permute(
                        2, 0, 1).reshape(shape=(len(batchX), self.data['sInChannels'], self.data['sInputDim'][0], self.data['sInputDim'][1]))
                
                # Resize input image
                # ? https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html#torch.nn.functional.pad
                heightPad = (224 - inTensor.shape[2]) // 2
                widthPad = (224 - inTensor.shape[3]) // 2
                padding = (heightPad, heightPad, widthPad, widthPad)
                inTensor = nn.functional.pad(
                    inTensor, padding, 'constant', 0)

                y_predTotal.extend(self.forward(inTensor.cuda()).cpu().max(dim=1)[1])

            # y_pred = self.forward(torch.FloatTensor(
            #     np.array(X)).to(torch.device('cuda:0')))    # X should be from the TESTING SET
            # calculate the testing loss
            # train_loss = self.loss_function(y_pred, y_true)   #? Later stuff

        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        # return y_pred.max(dim=1)[1]
        return y_predTotal

    def run(self):
        #! Visualize the architecture
        tempX = self.data['train']['X'][0]
        boardGraphInput = torch.FloatTensor(np.array(tempX))

        if self.data['sDataName'] == 'MNIST':
            boardGraphInput = boardGraphInput.reshape(shape=(1, self.data['sInChannels'], self.data['sInputDim'][0], self.data['sInputDim'][1]))
        else:
            boardGraphInput = torch.FloatTensor(np.array(self.data['train']['X'])).permute(
                2, 0, 1).reshape(shape=(1, self.data['sInChannels'], self.data['sInputDim'][0], self.data['sInputDim'][1]))

        # Resize input image
        # ? https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html#torch.nn.functional.pad
        heightPad = (224 - boardGraphInput.shape[2]) // 2
        widthPad = (224 - boardGraphInput.shape[3]) // 2
        padding = (heightPad, heightPad, widthPad, widthPad)
        boardGraphInput = nn.functional.pad(
            boardGraphInput, padding, 'constant', 0)

        self.writer.add_graph(self, boardGraphInput.cuda())

        #! Actual run
        print('method running...')
        print('--start training...')
        self.trainModel(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])

        # print(f'{pred_y} and the length {len(pred_y)} also length of each')
        # ALso for tensor
        return {'pred_y': pred_y, 'true_y': self.data['test']['y'],
                'loss': self.lossList}
