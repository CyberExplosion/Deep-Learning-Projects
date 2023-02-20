'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import pandas as pd
import pickle

#######################  : FINISH the dataset loader. This is now a dataset class
class Dataset_Loader(dataset):
    data = None
    # dataset_source_folder_path = u'P3/data/'

    def __init__(self, dName=None, dDescription=None, sDataset="MNIST"):
        super().__init__(dName, dDescription)
        self.dataName = sDataset


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