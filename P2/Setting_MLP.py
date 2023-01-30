'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np

class Setting_MLP(setting):
    randSeed = None
    kfold = None
    testSize = None

    # also use device to convert all to GPU tensor

    def __init__(self, sName=None, sDescription=None, sRandSeed=47, sKFold=5, sTestSize=0.33):
        super().__init__(sName, sDescription)
        self.randSeed = sRandSeed
        self.kfold = sKFold
        self.testSize = sTestSize
    
    def load_run_save_evaluate(self):
        # load dataset
        loaded_data = self.dataset.load()   # From the dataset Loader

        X_train, X_validate, y_train, y_validate = train_test_split(loaded_data['X'], loaded_data['y'], 
                                                                    test_size = self.testSize,
                                                                    random_state=self.randSeed)

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'validate': {'X': X_validate, 'y': y_validate}}
        learned_result = self.method.run()  # From the Method MLP
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()
            
        self.evaluate.data = learned_result
        
        return self.evaluate.evaluate()

        