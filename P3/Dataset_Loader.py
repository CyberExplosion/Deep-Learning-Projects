'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import pandas as pd
import pickle


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = u'P2/data/MNIST'
    # test_set_path = u'P2/data/test.csv'

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self) -> dict:
        print('loading data...')

        with open('./data/MNIST', 'rb') as f:
            data = pickle.load(f)    

        XTrain = data
        yTrain = []
        XTest = []
        yTest = []
        dt = pd.read_csv(self.dataset_source_folder_path, header=None)
        XTrain = dt.iloc[:,1::]
        yTrain = dt.iloc[:,0]

        testDt = pd.read_csv(self.test_set_path, header=None)
        XTest = testDt.iloc[:, 1::]
        yTest = testDt.iloc[:, 0] 


        return {
            'train': {
                'X': XTrain,
                'y': yTrain
            },
            'test': {
                'X': XTest,
                'y': yTest
            }
        }