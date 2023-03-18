'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from Dataset_Loader_Node_Classification import Dataset_Loader
from Result_Saver import Result_Saver
from Evaluate_Accuracy import Evaluate_Accuracy
from Method_Classification import Method_Classification
import numpy as np
import torch
import random

class Settings(setting):
    randSeed = None
    kfold = None
    testSize = None

    # also use device to convert all to GPU tensor

    def __init__(self, sName=None, sDescription=None, sRandSeed=47, sKFold=None, sDataset = "cora", sUseSave=True):
        super().__init__(sName, sDescription)
        self.trainParam = {}
        self.dataName = sDataset # or ORL or CIFAR
        self.useSave = sUseSave


        self.trainParam['sRandSeed'] = sRandSeed
        self.trainParam['kfold'] = sKFold

        # set seed
        np.random.seed(sRandSeed)
        random.seed(sRandSeed)
        torch.manual_seed(sRandSeed)
        torch.cuda.manual_seed(sRandSeed)
        torch.cuda.manual_seed_all(sRandSeed)

        if self.dataName == "cora":
            self.trainParam['sDataName'] = 'cora'
            self.trainParam['sInputDim'] = 1433 # Num features
            self.trainParam['sOutputDim'] = 7   # num classes
            self.prepare(
                sDataset=Dataset_Loader(dDataset=self.dataName),
                sMethod=Method_Classification(sData=self.trainParam),
                sEvaluate=Evaluate_Accuracy(),
                sResult=Result_Saver()
            )
        elif self.dataName == "citeseer":
            self.trainParam['sDataName'] = 'citeseer'
            self.trainParam['sInputDim'] = 3703
            self.trainParam['sOutputDim'] = 6
            self.prepare(
                sDataset=Dataset_Loader(dDataset=self.dataName),
                sMethod=Method_Classification(sData=self.trainParam),
                sEvaluate=Evaluate_Accuracy(),
                sResult=Result_Saver()
            )
        elif self.dataName == "pubmed":
            self.trainParam['sDataName'] = 'pubmed'
            self.trainParam['sInputDim'] = 500
            self.trainParam['sOutputDim'] = 3
            self.prepare(
                sDataset=Dataset_Loader(dDataset=self.dataName),
                sMethod=Method_Classification(sData=self.trainParam),
                sEvaluate=Evaluate_Accuracy(),
                sResult=Result_Saver()
            )
        
    
    def load_run_save_evaluate(self):
        # load dataset
        if self.useSave:
            print('Use saved processed data...')
            loaded_data = self.dataset.load_savedData()
        else:
            loaded_data = self.dataset.load()   # From the dataset Loader

        # Prepare train parameters
        self.method.data.update(loaded_data)
        self.method.data.update(self.trainParam)
            
        # run MethodModule
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = loaded_data
        self.result.fold_count = self.kfold
        self.result.model_res_name = self.method.model_res_name

        self.evaluate.data = learned_result
        self.evaluate.printOveralPerformance()

        self.result.data['acc'] = self.evaluate.evaluate()
        self.result.saveModel(self.method)


        return self.result.data['acc']
