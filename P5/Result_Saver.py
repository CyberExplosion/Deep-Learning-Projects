'''
Concrete ResultModule class for a specific experiment ResultModule output
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.result import result
import pickle
import torch.nn as nn
import torch


class Result_Saver(result):
    data = None
    fold_count = None
    result_destination_folder_path = 'P5/result/'
    result_destination_file_name = 'runLog'
    model_res_name = None
    def saveModel(self, model: nn.Module):
        print('saving models...')
        torch.save(model.state_dict(), self.result_destination_folder_path + self.model_res_name + '.pt')

    def save(self):
        print('saving results...')
        f = open(self.result_destination_folder_path +
                 self.result_destination_file_name + '_' + str(self.fold_count), 'wb')
        pickle.dump(self.data, f)
        f.close()
