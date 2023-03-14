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
from tqdm import trange
import torch_geometric.nn as geonn

BERTMAX_LEN = 512
class Method_Classification(method, nn.Module):
    data = {
        'sLossFunction': 'CrossEntropy',
        'sOptimizer': 'ADAM',
        'sInputDim': 768, # BERT
        'sOutputDim': 2,    # Binary classification
        'sLearningRate': 1e-6,
        'sMomentum': 0.9,
        'sMaxEpoch': 7,  # ! CHANGE LATER
        'sBatchSize': 2,  # Lower than 4000 is required
        'sRandSeed': 47
    }

    def __init__(self, mName='', mDescription='', sData=None):

        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        if sData:
            for k, v in sData.items():
                self.data[k] = v

        self.model_res_name = 'gcn_models'

        for k, v in self.data.items():
            if k != 'train' and k != 'test':
                self.model_res_name += f'_{k}:{v}'

        self.writer = SummaryWriter(
            comment=self.model_res_name)
        
        self.inputLayer = OrderedDict([
            ('gcnconv_layer_1', geonn.GCNConv(in_channels=self.data['sInputDim'], out_channels=50)),
            ('activation_1', nn.ReLU()),
            ('gcnconv_layer_2', geonn.GCNConv(in_channels=50, out_channels=self.data['sOutputDim'])),
        ])
        self.outputLayer = OrderedDict()

        # do not use softmax if we have nn.CrossEntropyLoss base on the PyTorch documents
        # ? https://stackoverflow.com/questions/55675345/should-i-use-softmax-as-output-when-using-cross-entropy-loss-in-pytorch
        if self.data['sLossFunction'] != 'CrossEntropy':
            self.outputLayer['output'] = nn.Softmax(dim=1)
        else:
            self.outputLayer['output'] = nn.Sigmoid()

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
         #! Need to be different if passing GCN to other layers
        for name, func in self.layers.items():
            out = func(out)
        return out

    def trainModel(self, X, y):
        # #!!!! Debugging 
        # torch.autograd.set_detect_anomaly(True)

        # Turn on train mode for all layers and prepare data
        self.training = True
        optimizer = self.optimizer
        loss_function = self.lossFunction

        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')


        for epoch in trange(self.data['sMaxEpoch'], desc='Training epochs'):
            permutation = torch.randperm(len(X))    # random order of batches
            for i in trange(0, len(X), self.data['sBatchSize'], desc=f'Batch progression at epoch {epoch}'):
                indices = permutation[i:i+self.data['sBatchSize']]
                batchX, batchY = [X[i] for i in indices], [y[i] for i in indices]   # batches
                # TODO: Have to generate batch edge that only matters in this batch

                batchXTensor = torch.stack(self.embeddingBatchToEntry(batchX))

                # ! Begin forward here
                fold_pred = self.forward(batchXTensor.cuda())
                fold_true = torch.LongTensor(np.array(batchY)).cuda()

                # calculate the training loss
                train_loss = loss_function(fold_pred, fold_true)
                optimizer.zero_grad()
                train_loss.backward()

                optimizer.step()

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
                if 'bertModel' not in name:
                    self.writer.add_histogram(name, weight, epoch)
                    self.writer.add_histogram(f'{name}.grad', weight.grad, epoch)

    def test(self, X):
        # Set to test mode
        y_predTotal = []
        self.training = False
        with torch.no_grad():
            for i in trange(0, len(X), self.data['sBatchSize'], desc='Test data batch progress'):
                batchX = X[i:i+self.data['sBatchSize']]
                embeddedTestBatch = self.embeddingBatchToEntry(batchX)
                inTensor = torch.stack(embeddedTestBatch).cuda()
                y_predTotal.extend(self.forward(inTensor).cpu().max(dim=1)[1])

        return y_predTotal

    def getEdgesThatMatter(self, batchX:list, edgeList:list):
        '''
        Use when creating batches of input

        Parameters:
        ---------------
        batchX: list
            The batch input (all are indices) of node ID
        edgeList: list
            The list of all edges. Each entry in the list in the form of [index cited, index cited from]
        '''
        edgeMatterList = []
        # print(f'The value of batch {batchX}')
        # print(f'The value in edgeList {edgeList}')

        # TODO: Count how many [0, 8] show up in edge List
        print(f'Count: {edgeList.count([0, 8])}')   #! 200 count ??????
        return
        
        for idx in batchX:
            # TODO: Find out why too many repitition in linksWithThisIdx
            print(f'Idx value in this loop: {idx}')
            linksWithThisIdx = [e for e in edgeList if e[0] == idx] # ! ONLY get the cited
            if linksWithThisIdx:
                print(f'at idx {idx}, the linksWIththisIDx is: {linksWithThisIdx}')
                edgeMatterList.extend(linksWithThisIdx)
        return edgeMatterList

    def run(self):
        #! Visualize the architecture
        # inputBatch = torch.stack(self.embeddingBatchToEntry(self.data['train']['X'][0:self.data['sBatchSize']])).cuda()
        # print(self.data['graph']['X'][0:self.data['sBatchSize']])
        inputBatch = torch.tensor(self.data['graph']['X'][0:self.data['sBatchSize']])
        
        batchIndexList = self.data['train_test']['idx_train'][0:self.data['sBatchSize']]
        print(f"The batch index list: {batchIndexList}")
        # TODO: USE THE WHOLE edge list and filter out the one we need during training - because of batches
        # print(f"Shape of the features: {self.data['graph']['X'].shape}")
        # print(f"Shape of the features element: {self.data['graph']['X'][0].shape}")
        # print(f"Shape of the features element: {np.squeeze(np.asarray(self.data['graph']['X'][0])).shape}")
        # print(f"Type of the features element: {type(np.squeeze(np.asarray(self.data['graph']['X'][0])))}")

        # print(f"Testing one element: {self.data['graph']['X'][0].tolist()}. The type: {type(self.data['graph']['X'][0].tolist())}")
        # print(f"The shape of object: {len(self.data['graph']['X'][0].tolist()[0])}")

        inputBatchList = [np.squeeze(np.asarray(self.data['graph']['X'][index])) for index in batchIndexList]
        edgeList = self.getEdgesThatMatter(batchIndexList, self.data['graph']['edge'])
        edgeList = torch.tensor(edgeList).cuda()
        
        # print(f'value edge list: {edgeList}')
        # print(f'Shape of edge list: {edgeList.shape}')
        return
        
        inputBatch = inputBatch.cuda()
        print(f'value: {inputBatch}')
        print(f'Shape of input: {inputBatch.shape}')



        self.writer.add_graph(self, input_to_model=(inputBatch, edgeList))

        #! Actual run
        print('method running...')
        print('--network status--')
        summary(self, 
                input_size=(self.data['sBatchSize'], 512, 768)
                )
        print('--start training...')
        self.trainModel(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])

        # print(f'{pred_y} and the length {len(pred_y)} also length of each')
        # ALso for tensor
        return {'pred_y': pred_y, 'true_y': self.data['test']['y'],
                'loss': self.lossList}