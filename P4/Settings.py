'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from Dataset_Loader import Dataset_Loader
from Method_Classification import Method_Classification
from Result_Saver import Result_Saver
from Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import random
import torch

class Settings(setting):
    randSeed = None
    kfold = None
    testSize = None

    # also use device to convert all to GPU tensor

    def __init__(self, sName=None, sDescription=None, sRandSeed=47, sKFold=None, sTask="text_classification", sUseSave=True):
        super().__init__(sName, sDescription)
        self.trainParam = {}
        self.task = sTask # or ORL or CIFAR
        self.useSaved = sUseSave

        # set seed
        np.random.seed(sRandSeed)
        random.seed(sRandSeed)
        torch.manual_seed(sRandSeed)
        torch.cuda.manual_seed(sRandSeed)
        torch.cuda.manual_seed_all(sRandSeed)


        self.trainParam['kfold'] = sKFold
        self.trainParam['task'] = sTask

        self.prepare(
            sDataset=Dataset_Loader(task=self.task),
            sMethod=Method_Classification(),
            sEvaluate=Evaluate_Accuracy(),
            sResult=Result_Saver()
        )

    
    def load_run_save_evaluate(self):
        # load dataset
        print(f'value of saved: {self.useSaved}')
        if self.useSaved:
            print('Using saved embedded tensor')
            loaded_data = self.dataset.loadTokenizedData()  # From the dataset Loader
        else:
            loaded_data = self.dataset.tokenize(save=True)


        # ! Remember to shuffle the loaded data when train

        # Prepare train parameters
        self.method.data.update(loaded_data)
        self.method.data.update(self.trainParam)

            
        # run MethodModule
        learned_result = self.method.run()  # From the Method MLP
            
        # save raw ResultModule
        # self.result.data = learned_result
        self.result.data = loaded_data
        self.result.fold_count = self.kfold
        # self.result.data.update(self.method.data)
        self.result.model_res_name = self.method.model_res_name

        self.evaluate.data = learned_result
        self.evaluate.printOveralPerformance()
        self.evaluate.plotLossGraph()

        self.result.data['acc'] = self.evaluate.evaluate()
        # self.result.save()
        self.result.saveModel(self.method)


        return self.result.data['acc']

        