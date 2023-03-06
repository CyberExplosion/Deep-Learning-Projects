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
from transformers import BertModel
from tqdm import trange

# class extract_tensorLTSM(nn.Module):
#     def forward(self,x):
#         # Output shape (batch, features, hidden)
#         tensor, _ = x
#         # Reshape shape (batch, hidden)
#         print(f'Value of the tensor: {tensor}')
#         ten = tensor[:, -1, :]
#         flat = nn.Flatten()
#         ten = flat(ten)
#         return ten

class Method_Generation(method, nn.Module):
    data = {
        'sLossFunction': 'CrossEntropy',
        'sOptimizer': 'ADAM',
        'sNumWordInput': 3,
        'sNumSequence': 18585,
        'sHiddenSize': 50,
        'sGradClipAmt': 0.5,
        'sDropout': 0.5,
        'sVocabSize': 4368,
        'sLearningRate': 1e-3,
        'sMomentum': 0.9,
        'sMaxEpoch': 500,  # ! CHANGE LATER
        'sRandSeed': 47
    }

    def __init__(self, mName='', mDescription='', sData=None):

        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        if sData:
            for k, v in sData.items():
                self.data[k] = v

        self.model_res_name = '_generation'

        for k, v in self.data.items():
            if k != 'train' and k != 'test':
                self.model_res_name += f'_{k}:{v}'
        self.writer = SummaryWriter(
            comment=self.model_res_name)
        
        self.inputLayer = OrderedDict([
            ('embedding_layer', nn.Embedding(num_embeddings=self.data['sNumSequence'],
                                             embedding_dim=self.data['sVocabSize']))
        ])
        self.hiddenLayer = OrderedDict([
            ('lstm_layer_1', nn.LSTM(input_size=self.data['sVocabSize'],
                                     hidden_size=self.data['sHiddenSize'])),
        ])
        self.outputLayer = OrderedDict([
            ('flatten_layer', nn.Flatten()),   # No batch
            ('linear_layer_2', nn.Linear(in_features=150, out_features=self.data['sVocabSize']))
        ])

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
        res.update(self.hiddenLayer)
        res.update(self.outputLayer)
        return res

    def forward(self, x):
        '''Forward propagation'''
        out = x
        for name, func in self.layers.items():
            if 'lstm' in name:
                out = func(out)[0]
            else:
                out = func(out)
        return out

    def trainModel(self, X, y):

        # Turn on train mode for all layers and prepare data
        self.training = True
        optimizer = self.optimizer
        loss_function = self.lossFunction
        accuracy_evaluator = Evaluate_Accuracy('training evaluator')

        for epoch in trange(self.data['sMaxEpoch'], desc='Training epochs progression'):
            # ! Begin forward here
            fold_pred = self.forward(X.cuda())
            # fold_true = torch.LongTensor(np.array(y)).cuda()
            fold_true = y.cuda()

            # print(f'The target label of first: {fold_true[0]}')

            # print(f'len of the target batch: {fold_true.shape}')
            # print(f'len of the predicted batch: {fold_pred.shape}')
            
            # calculate the training loss
            if self.data['sLossFunction'] == 'CrossEntropy':
                train_loss = loss_function(fold_pred, fold_true)
            else:
                train_loss = loss_function(fold_pred, fold_true)
                
            optimizer.zero_grad()
            train_loss.backward()

            # TODO: Potential modification
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.data['sGradClipAmt'])

            optimizer.step()

            if epoch % 50 == 0:
                # The y_pred.max(1)[1] return the indices of max value on each row of a tensor (the y_pred is a tensor)
                accuracy_evaluator.data = {
                    'true_y': fold_true.cpu(), 'pred_y': fold_pred.cpu().max(dim=1)[1]}
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
                # print(f'name: {name}, weight: {weight}, at this epoch: {epoch}')
                self.writer.add_histogram(name, weight, epoch)
                self.writer.add_histogram(f'{name}.grad', weight.grad, epoch)

    def test(self, X):
        # Set to test mode
        y_predTotal = []
        self.training = False
        with torch.no_grad():
            # TODO: Change into using the whole thing instead of using batch
            inTensor = X.cuda()
            y_predTotal = self.forward(inTensor).cpu().max(dim=1)[1]

        return y_predTotal

    def run(self):
        #! Visualize the architecture
        testIn = torch.randint(low=0, high=100, size=(self.data['sNumSequence'], self.data['sNumWordInput'])).cuda()
        self.writer.add_graph(self, testIn)

        #! Actual run
        print('method running...')
        print('--network status--')
        summary(self, input_data=testIn)
        
        print('--start training...')
        self.trainModel(self.data['X'], self.data['y'])
       
        print('--start testing...')
        pred_y = self.test(self.data['X'])

        return {'pred_y': pred_y, 'true_y': self.data['y'],
                'loss': self.lossList}
