'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = 'P2/data/'
    dataset_source_file_name = 'train.csv'
    test_set_path = 'P2/data/test.csv'

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self) -> dict:
        print('loading data...')
        XTrain = []
        yTrain = []
        XTest = []
        yTest = []
        fTrain = open(self.dataset_source_folder_path +
                      self.dataset_source_file_name, 'r')
        for line in fTrain:
            line = line.strip('\n')
            elements = [int(i) for i in line.split(',')]
            XTrain.append(elements[1::])  # change the input dimension
            yTrain.append(elements[0])
        fTrain.close()
        fTest = open(self.test_set_path, 'r')
        for line in fTest:
            line = line.strip('\n')
            elements = [int(i) for i in line.split(',')]
            XTest.append(elements[1::]) # change the input dimension
            yTest.append(elements[0])
        fTest.close()

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