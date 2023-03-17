'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import torch

class Evaluate_Accuracy(evaluate):
    data = None
    
    def printOveralPerformance(self):
        print('run performace metrics: ')
        # print(f"The predicted: {self.data['pred_y']}")
        # print(f"and the true: {self.data['true_y']}")
        # print(f"Shape of the prediction: {self.data['pred_y'].shape} and the true: {self.data['true_y'].shape}")
        print(classification_report(y_pred=self.data['pred_y'], y_true=self.data['true_y']))

    def evaluate(self):
        return accuracy_score(self.data['true_y'], self.data['pred_y'])
        