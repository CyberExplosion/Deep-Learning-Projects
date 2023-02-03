'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import pandas as pd


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = u'P2/data/'
    dataset_source_file_name = u'train.csv'
    test_set_path = u'P2/data/test.csv'

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self) -> dict:
        print('loading data...')
        XTrain = []
        yTrain = []
        XTest = []
        yTest = []
        dt = pd.read_csv(self.dataset_source_folder_path + self.dataset_source_file_name, header=None)
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