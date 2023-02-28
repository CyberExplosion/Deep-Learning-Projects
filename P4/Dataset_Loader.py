# Custom made for BERT

from code.base_class.dataset import dataset
import torch
from transformers import BertTokenizer
from pathlib import Path
import re
import pickle

class Dataset_Loader(dataset):
    data = {}

    def __init__(self, dName=None, dDescription=None, task='text_classification'):
        super().__init__(dName)
        # self.task = task
        self.data['train'] = []
        self.data['test'] = []
        self.dataFolder = Path('P4', 'data', task)
        self.task = task
        self.tensorData = {}
        self.tensorData['train'] = []
        self.tensorData['test'] = []

    def cleanTrainData(self) -> list:
        cleanedData = []
        CLEANHTML = re.compile('<.*?>')

        for entry in self.data['train']:
            # ! Hope the data doesn't contain heavy html tags or else it wouldn't work
            text = re.sub(CLEANHTML, '', entry[0])
            cleanedData.append((text, entry[1]))
        return cleanedData

    # Warning this would not work for BERT model - Only for RNN
    def encodeForRNN(self) -> dict:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        BERT_MAX_LENGTH = 512
        for i, (text, label) in enumerate(self.data['train']):
            out = tokenizer(text, padding='max_length', max_length=BERT_MAX_LENGTH, add_special_tokens=True)    
            tokenized = out['input_ids']
            tokenType = out['token_type_ids']
            attention = out['attention_mask']
            truncated = {
                'input_ids': tokenized[0:1] + tokenized[-(BERT_MAX_LENGTH-1):],    # keep special start and end symbol
                'token_type_ids': tokenType[0:1] + tokenType[-(BERT_MAX_LENGTH-1):],
                'attention_mask': attention[0:1] + attention[-(BERT_MAX_LENGTH-1):],
            }
            self.data['train'][i] = (torch.tensor(tokenized), label)

        for i, (text, label) in enumerate(self.data['test']):
            out = tokenizer(text, padding='max_length', max_length=BERT_MAX_LENGTH, add_special_tokens=True)    
            tokenized = out['input_ids']
            tokenType = out['token_type_ids']
            attention = out['attention_mask']
            truncated = {
                'input_ids': tokenized[0:1] + tokenized[-(BERT_MAX_LENGTH-1):],    # keep special start and end symbol
                'token_type_ids': tokenType[0:1] + tokenType[-(BERT_MAX_LENGTH-1):],
                'attention_mask': attention[0:1] + attention[-(BERT_MAX_LENGTH-1):],
            }
            self.data['test'][i] = (torch.tensor(tokenized), label)               
            
        return self.data

            
    def load(self, save=False) -> dict:
        # * 1 is for positive, 0 is negative
        print('loading data....')
        dataPath = self.dataFolder

        for p in Path(dataPath, 'train', 'pos').glob('*.txt'):  # 1 is pos and 0 is neg
            entry = p.read_text(encoding='utf8')
            self.data['train'].append((entry, 1))
        for p in Path(dataPath, 'train', 'neg').glob('*.txt'):  # 1 is pos and 0 is neg
            entry = p.read_text(encoding='utf8')
            self.data['train'].append((entry, 0))

        for p in Path(dataPath, 'test', 'pos').glob('*.txt'):  # 1 is pos and 0 is neg
            entry = p.read_text(encoding='utf8')
            self.data['test'].append((entry, 1))
        for p in Path(dataPath, 'test', 'neg').glob('*.txt'):  # 1 is pos and 0 is neg
            entry = p.read_text(encoding='utf8')
            self.data['test'].append((entry, 0))

        # Clean the training data
        self.data['train'] = self.cleanTrainData()
        # Convert data to tensor
        self.data = self.encodeForRNN()

        # Save the conversion
        if save:
            with open(f'P4/saved/{self.task}-dataInTensor', 'wb') as handle:
                pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return self.data

test = Dataset_Loader()
res = test.load(save=True)
print(res['train'][0])
