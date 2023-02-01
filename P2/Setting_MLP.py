'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np
from Dataset_Loader import Dataset_Loader
from Method_MLP import Method_MLP
from Result_Saver import Result_Saver
from Evaluate_Accuracy import Evaluate_Accuracy
import json

class Setting_MLP(setting):
    randSeed = None
    kfold = None
    testSize = None

    # also use device to convert all to GPU tensor

    def __init__(self, sName=None, sDescription=None, sRandSeed=47, sKFold=5, extraSettingsPath=None):
        super().__init__(sName, sDescription)
        self.trainParam = {}
        if extraSettingsPath:
            with open(extraSettingsPath) as f:
                extraConfigs = f.read()
                js = json.loads(extraConfigs)
                self.trainParam.update(js)

        self.trainParam['randSeed'] = sRandSeed
        self.trainParam['kfold'] = sKFold

        self.prepare(
            sDataset=Dataset_Loader(),
            sMethod=Method_MLP(sData=self.trainParam),
            sEvaluate=Evaluate_Accuracy(),
            sResult=Result_Saver()
        )

    
    def load_run_save_evaluate(self):
        # load dataset
        loaded_data = self.dataset.load()   # From the dataset Loader

        # X_train, X_validate, y_train, y_validate = train_test_split(loaded_data['X'], loaded_data['y'],
        #                          
        #                                    random_state=self.randSeed)
        
        # run MethodModule
        self.method.data.update(loaded_data)
        self.method.data.update(self.trainParam)

        learned_result = self.method.run()  # From the Method MLP
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.data['sRandSeed'] = self.randSeed
        self.result.fold_count = self.kfold

        self.result.save()
            
        self.evaluate.data = learned_result
        
        return self.evaluate.evaluate()

        