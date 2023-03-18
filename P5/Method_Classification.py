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
from tqdm import trange, tqdm
import torch_geometric.nn as geonn
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

BERTMAX_LEN = 512
class Method_Classification(method, nn.Module):
    data = {
        'sLossFunction': 'CrossEntropy',
        'sOptimizer': 'ADAM',
        'sInputDim': 1433, # BERT
        'sOutputDim': 7,    # Binary classification
        'sLearningRate': 1e-4,
        'sMomentum': 0.9,
        'sMaxEpoch': 2000,  # ! CHANGE LATER
        'sRandSeed': 31,
        'sKFold': 1
    }

    def __init__(self, mName='', mDescription='', sData=None):

        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        if sData:
            for k, v in sData.items():
                self.data[k] = v

        self.model_res_name = 'gcn_models'

        for k, v in self.data.items():
            if k != 'graph' and k != 'train_test':
                self.model_res_name += f'_{k}:{v}'

        self.writer = SummaryWriter(
            comment=self.model_res_name)
        
        self.layers = geonn.Sequential('x, edge_index',[
            (geonn.GCNConv(in_channels=self.data['sInputDim'], out_channels=50), 'x, edge_index -> x'),
            nn.ReLU(),
            # (geonn.GCNConv(in_channels=100, out_channels=50), 'x, edge_index -> x'),
            # nn.ReLU(),
            (geonn.GCNConv(in_channels=50, out_channels=self.data['sOutputDim']), 'x, edge_index -> x'),
            nn.Sigmoid(),
            # nn.Linear(in_features=16, out_features=self.data['sOutputDim']),
            # nn.ReLU()
        ])


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
        self.trainMask, self.testMask = [], []

        self.cuda()   # ! Try without cuda for graph


    def forward(self, x, edge_index):
        '''Forward propagation'''
        out = self.layers(x, edge_index)
        return out

    def trainModel(self, X, edge_index, y):
        # #!!!! Debugging 
        # Turn on train mode for all layers and prepare data
        optimizer = self.optimizer
        loss_function = self.lossFunction

        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        # cv = KFold(n_splits=self.data['sKFold'], shuffle=True, random_state=self.data['sRandSeed'])

        # ! Try doing no fold

        # for fold, (train_idx, val_idx) in enumerate(cv.split(self.data['train_test']['idx_train'])):
        self.training = True
        # print(f"value of the idx train: {self.data['train_test']['idx_train']}")
        
        # trainMask = self.createMaskForGraphData(train_idx)
        trainMask = self.createMaskForGraphData(self.data['train_test']['idx_train'])

        for epoch in trange(self.data['sMaxEpoch'], desc='Training epochs'):
            # ! Begin forward here
            fold_pred = self.forward(X.cuda(), edge_index.cuda())
            fold_true = torch.LongTensor(np.array(y)).cuda()

            # ! only calculated loss using the training label
            train_loss = loss_function(fold_pred[trainMask], fold_true[trainMask])
            optimizer.zero_grad()
            train_loss.backward()

            optimizer.step()

            # The y_pred.max(1)[1] return the indices of max value on each row of a tensor (the y_pred is a tensor)
            if epoch % 100 == 0:
                accuracy_evaluator.data = {
                    'true_y': fold_true.cpu(), 'pred_y': fold_pred.cpu().max(dim=1)[1]}
                acc = accuracy_evaluator.evaluate()
                loss = train_loss.item()
                # print('Fold:', fold,'Epoch:', epoch, 'Accuracy:',
                #         acc, 'Loss:', loss)
                print('Epoch:', epoch, 'Accuracy:',
                        acc, 'Loss:', loss)

            # Record data for ploting
            self.lossList.append(loss)

            # Record data for tensorboard
            self.writer.add_scalar('Training Loss', train_loss, epoch)
            self.writer.add_scalar('Accuracy', acc, epoch)

            # Check learning progress
            for name, weight in self.named_parameters():
                if 'bertModel' not in name:
                    self.writer.add_histogram(name, weight, epoch)
                    self.writer.add_histogram(f'{name}.grad', weight.grad, epoch)

            # # Testing the fold
            # self.training = False
            # testMask = self.createMaskForGraphData(val_idx)
            # with torch.no_grad():
            #     rawPred = self.forward(X.cuda(), edge_index.cuda())
            #     testPred = rawPred[testMask].cpu().max(dim=1)[1]     # Only get the test label, no need the train label

            # print(f"The predicted: {testPred}")
            # print(f"and the true: {self.data['graph']['y'][testMask]}")
            # print(f"Shape of the prediction: {testPred.shape} and the true: {self.data['graph']['y'][testMask].shape}")
                
            # print(f"Fold {fold} testing acc: {accuracy_score(self.data['graph']['y'][testMask], testPred)}")


    def test(self, X, edge_index):
        # Set to test mode
        self.training = False
        testMask = self.createMaskForGraphData(self.data['train_test']['idx_test'])
        with torch.no_grad():
            rawPred = self.forward(X.cuda(), edge_index.cuda())
            testPred = rawPred[testMask].cpu().max(dim=1)[1]     # Only get the test label, no need the train label

        return testPred


    def createMaskForGraphData(self, idxToUse:list) -> list:
        '''
        Create a mask Tensor to hide all the labels that not use

        Use in creating mask for fold
        '''
        numNodes = len(self.data['graph']['X'])

        mask = torch.zeros(numNodes, dtype=torch.bool)
        mask[idxToUse] = True # mark true for only nodes that in the training set

        return mask


    def run(self):
        #! Visualize the architecture

        #  USE THE WHOLE edge list and filter out the one we need during training - because of batches
        # ! CANNOT do it in batches, because you have to include the neighbor node in that edges also in the features tensor

        # TODO: The Right way is to pass in all edges as always, but only use certain number of nodes during training
        # TODO: WE USING transductive learning (the training know the whole graph), but it doesn't know the label of the test set => Mask the training and testing

        # self.createMaskForGraphData()
        self.writer.add_graph(self, input_to_model=(self.data['graph']['X'].cuda(), self.data['graph']['edge'].cuda()))

    
        #! Actual run
        print('method running...')
        print('--network status--')
        summary(self, 
                input_size=[self.data['graph']['X'].shape, self.data['graph']['edge'].shape],
                dtypes=[torch.float, torch.long] 
                )
        print('--start training...')
        self.trainModel(self.data['graph']['X'], self.data['graph']['edge'], self.data['graph']['y'])

        print('--start testing...')
        pred_y = self.test(self.data['graph']['X'], self.data['graph']['edge'])

        # print(f'{pred_y} and the length {len(pred_y)} also length of each')
        trueTestMaks = self.createMaskForGraphData(self.data['train_test']['idx_test'])

        # * Only care about the test mask
        return {'pred_y': pred_y, 'true_y': self.data['graph']['y'][trueTestMaks],
                'loss': self.lossList}