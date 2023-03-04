# Custom made for BERT

from code.base_class.dataset import dataset
import torch
from transformers import BertTokenizer, BertModel
from pathlib import Path
import re
import pickle
from tqdm import tqdm, trange
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

class Dataset_Loader(dataset):
    data = {}

    def __init__(self, dName=None, dDescription=None, task='test_data'):
        super().__init__(dName)
        self.processingData = {
            'train': [],
            'test': []
        }
        self.inputData = {
            'train': {
                'X': [],
                'y': []
            },
            'test': {
                'X': [],
                'y': []
            }
        }
        self.dataPath = Path('P4','data', task)     # ! Need to add P4 in front of the path
        self.picklePath = 'P4/saved'
        self.task = task
        self.bertBatchSize = 100

#     def saveTensorToPickle(self, tensorData: dict):
#         # Since it is a very big data (with 50000 reviews), save pickle with increment of 5000 review eachs
#         try:
#             with open(f'{self.picklePath}/{self.task}-pickleObjLen.pickle', 'rb') as handle:
#                 tensorLen = pickle.load(file=handle)
#         except FileNotFoundError:
#             print('file pickelObjLen.pickle not found. Creating count from begining')
#             tensorLen = {
#                 'train': {
#                     'X': 0,
#                     'y': 0,
#                 },
#                 'test': {
#                     'X': 0,
#                     'y': 0
#                 }
#             }
#             
#         if 'train' in tensorData.keys():
#             with open(f"{self.picklePath}/{self.task}-TrainTensorX.pickle", 'ab') as handle:
#                 for each in tensorData['train']['X']:
#                     pickle.dump(each, file=handle, protocol=pickle.HIGHEST_PROTOCOL)
#                     tensorLen['train']['X'] += 1
#             with open(f"{self.picklePath}/{self.task}-TrainTensorY.pickle", 'ab') as handle:
#                 for each in tensorData['train']['y']:
#                     pickle.dump(each, file=handle, protocol=pickle.HIGHEST_PROTOCOL)
#                     tensorLen['train']['y'] += 1
# 
#         if 'test' in tensorData.keys():
#             with open(f"{self.picklePath}/{self.task}-TestTensorX.pickle", 'ab') as handle:
#                 for each in tensorData['test']['X']:
#                     pickle.dump(each, file=handle, protocol=pickle.HIGHEST_PROTOCOL)
#                     tensorLen['test']['X'] += 1
#             with open(f"{self.picklePath}/{self.task}-TestTensorY.pickle", 'ab') as handle:
#                 for each in tensorData['test']['y']:
#                     pickle.dump(each, file=handle, protocol=pickle.HIGHEST_PROTOCOL)
#                     tensorLen['test']['y'] += 1
#         
#         with open(f'{self.picklePath}/{self.task}-pickleObjLen.pickle', 'ab') as handle:
#             pickle.dump(tensorLen, file=handle, protocol=pickle.HIGHEST_PROTOCOL)
        


    def loadTokenizedData(self) -> dict:
        with open(f'{self.picklePath}/{self.task}-tokenizedData.pickle', 'rb') as handle:
            tensorData = pickle.load(handle)
        return tensorData

    def tokenize(self, save=False) -> dict:
        # * 1 is for positive, 0 is negative
        print('loading data....')
        processedData = {
            'train': [],
            'test': []
        }

        for p in Path(self.dataPath, 'train', 'pos').glob('*.txt'):  # 1 is pos and 0 is neg
            entry = p.read_text(encoding='utf8')
            processedData['train'].append((entry, 1))
        for p in Path(self.dataPath, 'train', 'neg').glob('*.txt'):  # 1 is pos and 0 is neg
            entry = p.read_text(encoding='utf8')
            processedData['train'].append((entry, 0))

        for p in Path(self.dataPath, 'test', 'pos').glob('*.txt'):  # 1 is pos and 0 is neg
            entry = p.read_text(encoding='utf8')
            processedData['test'].append((entry, 1))
        for p in Path(self.dataPath, 'test', 'neg').glob('*.txt'):  # 1 is pos and 0 is neg
            entry = p.read_text(encoding='utf8')
            processedData['test'].append((entry, 0))

        # Clean the training data
        cleanedData = []
        CLEANHTML = re.compile('<.*?>')

        for entry in processedData['train']:
            # ! Hope the data doesn't contain heavy html tags or else it wouldn't work
            text = re.sub(CLEANHTML, '', entry[0])
            text = text.lower()
            cleanedData.append((text, entry[1]))

        # Tokenizing the data
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        BERT_MAX_LENGTH = 512
        inputData = {
            'train': {
                'X': [],
                'y': []
            },
            'test': {
                'X': [],
                'y': []
            }
        }

        for i, (text, label) in enumerate(tqdm(processedData['train'], desc='Passing train data through tokenizer')):
            out = tokenizer(text, padding='max_length', add_special_tokens=True)
            tokenized = out['input_ids']
            tokenType = out['token_type_ids']
            attention = out['attention_mask']
            truncated = {
                # keep special start and end symbol
                'input_ids': tokenized[0:1] + tokenized[-(BERT_MAX_LENGTH-1):],
                'token_type_ids': tokenType[0:1] + tokenType[-(BERT_MAX_LENGTH-1):],
                'attention_mask': attention[0:1] + attention[-(BERT_MAX_LENGTH-1):],
            }
            inputData['train']['X'].append(truncated)
            inputData['train']['y'].append(label)

        # ! SKIP the test data tokenizing for testing purposes
        for i, (text, label) in enumerate(tqdm(processedData['test'], desc='Passing test data through tokenizer')):
            out = tokenizer(text, padding='max_length',
                            max_length=BERT_MAX_LENGTH, add_special_tokens=True)
            tokenized = out['input_ids']
            tokenType = out['token_type_ids']
            attention = out['attention_mask']
            truncated = {
                'input_ids': tokenized[0:1] + tokenized[-(BERT_MAX_LENGTH-1):],
                'token_type_ids': tokenType[0:1] + tokenType[-(BERT_MAX_LENGTH-1):],
                'attention_mask': attention[0:1] + attention[-(BERT_MAX_LENGTH-1):],
            }
            inputData['test']['X'].append(truncated)
            inputData['test']['y'].append(label)

        if save:
            print('saving the tokenized data...')
            with open(f'{self.picklePath}/{self.task}-tokenizedData.pickle', 'wb') as handle:
                pickle.dump(inputData, file=handle)

        return inputData
    

# test = Dataset_Loader(task='test_data_big')
# res = test.tokenize(save=True)
# 
# print(len(res['train']['X']))
# print(res['train']['X'][4])
# print(res['train']['X'][4])
# 
# print('Using saved, should have same output:')
# loaded = test.loadTokenizedData()
# print(len(loaded['train']['X']))
# print(loaded['train']['X'][4])
# print(loaded['train']['X'][4])