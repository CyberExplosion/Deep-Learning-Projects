'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

class Evaluate_Accuracy(evaluate):
    data = None
    
    def printOveralPerformance(self):
        print('run performace metrics: ')
        print(classification_report(y_pred=self.data['pred_y'], y_true=self.data['true_y']))

    def evaluate(self):
        return accuracy_score(self.data['true_y'], self.data['pred_y'])
        