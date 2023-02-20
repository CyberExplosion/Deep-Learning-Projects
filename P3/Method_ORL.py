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
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


class Method_ORL(method, nn.Module):
    data = {
        # 'sInitMethod': 'kaiming',
        'sLossFunction': 'CrossEntropy',
        'sOptimizer': 'SGD',   #TODO: Try SGD - betteer for images
        'sInputDim': (28, 28),
        'sInChannels': 1,   # Color channel (number of filter)
        'sDropout': 0.5,
        'sOutputDim': 40,
        'sLearningRate': 1e-3,
        'sMomentum': 0.9,
        'sMaxEpoch': 400,  # ! CHANGE LATER
        'sBatchSize': 100  # Lower than 4000 is required
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
        torch.manual_seed(self.data['sRandSeed'])

        self.model_res_name = 'nn_models'

        for k, v in self.data.items():
            if k != 'train' and k != 'test':
                self.model_res_name += f'_{k}:{v}'
        self.writer = SummaryWriter(
            comment=self.model_res_name)
        
        # Resize the input with transform
        self.transform = transforms.Compose([
            transforms.Resize((min(self.data['sInputDim']), min(self.data['sInputDim'])))
        ])

        # ! Alexnet takes too long to train, use a simpler cnn
        # https://www.kaggle.com/code/maneeshsehagal/facial-recognition-cnn-maneeshsehagal/notebook
        self.inputLayer = OrderedDict([
            ('conv_layer_1', nn.Conv2d(in_channels=self.data['sInChannels'],
                            out_channels=36, kernel_size=7)), #86 out
            ('activation_func_1', nn.ReLU()),
            ('maxpool_layer_1', nn.MaxPool2d(kernel_size=2)),   #43 out #! MAX POOL stride is equal to the kernel size
        ])
        self.features = OrderedDict([
            ('conv_layer_2', nn.Conv2d(36, 54, kernel_size=5)), #81
            ('activation_func_2', nn.ReLU()),
            ('maxpool_layer_2', nn.MaxPool2d(kernel_size=2)),  #80

            # ('avg_pool_layer', nn.AdaptiveAvgPool2d((7, 7)))
        ])
        self.classifier = OrderedDict([
            ('transitional_flat_layer', nn.Flatten()),
            
            ('linear_layer_3', nn.Linear(19494, 2024)),
            ('activation_func_3', nn.ReLU()),
            ('dropout_layer_3', nn.Dropout(p=self.data['sDropout'])),

            ('linear_layer_4', nn.Linear(2024, 1024)),
            ('activation_func_4', nn.ReLU()),
            ('dropout_layer_4', nn.Dropout(p=self.data['sDropout'])),

            ('linear_layer_5', nn.Linear(1024, 512)),
            ('activation_func_5', nn.ReLU()),
            ('dropout_layer_5', nn.Dropout(p=self.data['sDropout'])),
        ])
        self.outputLayer = OrderedDict([
            ('linear_layer_6', nn.Linear(512, self.data['sOutputDim']))
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

                inTensor = torch.FloatTensor(np.array(batchX)).permute(0, 3, 1, 2)

            # Use the smaller size (width or height) to downsampling the picture to square
            # ? https://pytorch.org/docs/stable/generated/torch.ao.nn.quantized.functional.interpolate.html
                # smallerSize = min(inTensor.shape[-1], inTensor.shape[-2])
                # inTensor = nn.functional.interpolate(inTensor.float(), size=(smallerSize, smallerSize), mode='bilinear')
                # print(f'Shape input: {inTensor.shape}')
                inTensor = self.transform(inTensor)

                # ! Begin forward here
                fold_pred = self.forward(inTensor.cuda())

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
                # Resize input image
                # smallerSize = min(inTensor.shape[-1], inTensor.shape[-2])
                # inTensor = nn.functional.interpolate(inTensor, size=(smallerSize, smallerSize), mode='bilinear')
                inTensor = self.transform(inTensor)

                y_predTotal.extend(self.forward(inTensor.cuda()).cpu().max(dim=1)[1])

        return y_predTotal

    def run(self):
        #! Visualize the architecture
        tempX = self.data['train']['X'][0]
        boardGraphInput = torch.FloatTensor(np.array(tempX))

        boardGraphInput = boardGraphInput.permute(2, 0, 1).unsqueeze(dim=0)
        # Resize input image
        # smallerSize = min(boardGraphInput.shape[-1], boardGraphInput.shape[-2])
        # boardGraphInput = nn.functional.interpolate(boardGraphInput, size=(smallerSize, smallerSize), mode='bilinear')
        boardGraphInput = self.transform(boardGraphInput)
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
