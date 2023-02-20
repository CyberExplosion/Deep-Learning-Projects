'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from Dataset_Loader import Dataset_Loader
from Method_MNIST import Method_MNIST
from Result_Saver import Result_Saver
from Evaluate_Accuracy import Evaluate_Accuracy
from Method_ORL import Method_ORL
from Method_CIFAR import Method_CIFAR

class Settings(setting):
    randSeed = None
    kfold = None
    testSize = None

    # also use device to convert all to GPU tensor

    def __init__(self, sName=None, sDescription=None, sRandSeed=47, sKFold=None, sDataset = "MNIST"):
        super().__init__(sName, sDescription)
        self.trainParam = {}
        self.dataName = sDataset # or ORL or CIFAR


        self.trainParam['sRandSeed'] = sRandSeed
        self.trainParam['kfold'] = sKFold


        if self.dataName == "MNIST":
            self.trainParam['sDataName'] = 'MNIST'
            self.trainParam['sInputDim'] = (28,28)
            self.trainParam['sInChannels'] = 1
            self.trainParam['sOutputDim'] = 10
            self.prepare(
                sDataset=Dataset_Loader(sDataset=self.dataName),
                sMethod=Method_MNIST(sData=self.trainParam),
                sEvaluate=Evaluate_Accuracy(),
                sResult=Result_Saver()
            )
        elif self.dataName == "ORL":
            self.trainParam['sDataName'] = 'ORL'
            self.trainParam['sInputDim'] = (112,92)
            self.trainParam['sInChannels'] = 3
            self.trainParam['sOutputDim'] = 40
            self.prepare(
                sDataset=Dataset_Loader(sDataset=self.dataName),
                sMethod=Method_ORL(sData=self.trainParam),
                sEvaluate=Evaluate_Accuracy(),
                sResult=Result_Saver()
            )
        elif self.dataName == "CIFAR":
            self.trainParam['sDataName'] = 'CIFAR'
            self.trainParam['sInputDim'] = (32,32)
            self.trainParam['sInChannels'] = 3
            self.trainParam['sOutputDim'] = 10
            self.prepare(
                sDataset=Dataset_Loader(sDataset=self.dataName),
                sMethod=Method_CIFAR(sData=self.trainParam),
                sEvaluate=Evaluate_Accuracy(),
                sResult=Result_Saver()
            )
        
    
    def load_run_save_evaluate(self):
        # load dataset
        loaded_data = self.dataset.load()   # From the dataset Loader

        # Prepare train parameters
        self.method.data.update(loaded_data)
        self.method.data.update(self.trainParam)

        #! ORL Labels starts at 1, which is incompatible with CUDA since it keeps indexing out of bound
        #! We minnus every labels by 1
        print(f'The data is {self.dataName}')
        if self.dataName == 'ORL':
            self.method.data['train']['y'] = [x - 1 for x in self.method.data['train']['y']]
            self.method.data['test']['y'] = [x - 1 for x in self.method.data['test']['y']]
            
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

        