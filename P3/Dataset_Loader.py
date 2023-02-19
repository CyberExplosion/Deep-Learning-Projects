'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import pandas as pd
import pickle
from torch.utils.data import Dataset

class Dataset_Loader(dataset, Dataset):
    data = None
    dataset_source_folder_path = u'P2/data/MNIST'

    def __init__(self, dName=None, dDescription=None, sDataset="MNIST", sType='train'):
        super().__init__(dName, dDescription)
        self.dataName = sDataset
        rawData = self.load()
        if sType == 'train':
            self.labels = rawData['train']['X']
            self.imgs = rawData['train']['y']
        else:
            self.labels = rawData['test']['X']
            self.imgs = rawData['test']['y']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.labels[idx]
        return image, label


    def load(self) -> dict:
        print('loading data...')

        with open(f'P3/data/{self.dataName}', 'rb') as f:
            data = pickle.load(f)    

        trainX, trainY = [], []
        for entries in data['train']:
            trainX.append(entries['image'])
            trainY.append(entries['label'])
        
        testX, testY = [], []
        for entries in data['test']:
            testX.append(entries['image'])
            testY.append(entries['label'])

        return {
            'train': {
                'X': trainX,
                'y': trainY
            },
            'test': {
                'X': testX,
                'y': testY
            }
        }
    
test = Dataset_Loader()