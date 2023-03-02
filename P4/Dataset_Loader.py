# Custom made for BERT

from code.base_class.dataset import dataset
import torch
from transformers import BertTokenizer, BertModel
from pathlib import Path
import re
import pickle


class Dataset_Loader(dataset):
    data = {}

    def __init__(self, dName=None, dDescription=None, task='text_classification'):
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
        self.dataFolder = Path('P4', 'data', task)
        self.task = task


    def load(self, save=False) -> dict:
        # * 1 is for positive, 0 is negative
        print('loading data....')
        processedData = {
            'train': [],
            'test': []
        }
        dataPath = 'P4/data/test_data'

        for p in Path(dataPath, 'train', 'pos').glob('*.txt'):  # 1 is pos and 0 is neg
            entry = p.read_text(encoding='utf8')
            processedData['train'].append((entry, 1))
        for p in Path(dataPath, 'train', 'neg').glob('*.txt'):  # 1 is pos and 0 is neg
            entry = p.read_text(encoding='utf8')
            processedData['train'].append((entry, 0))

        for p in Path(dataPath, 'test', 'pos').glob('*.txt'):  # 1 is pos and 0 is neg
            entry = p.read_text(encoding='utf8')
            processedData['test'].append((entry, 1))
        for p in Path(dataPath, 'test', 'neg').glob('*.txt'):  # 1 is pos and 0 is neg
            entry = p.read_text(encoding='utf8')
            processedData['test'].append((entry, 0))

        # Clean the training data
        cleanedData = []
        CLEANHTML = re.compile('<.*?>')

        for entry in processedData['train']:
            # ! Hope the data doesn't contain heavy html tags or else it wouldn't work
            text = re.sub(CLEANHTML, '', entry[0])
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

        for i, (text, label) in enumerate(processedData['train']):
            out = tokenizer(text, padding='max_length',
                            max_length=BERT_MAX_LENGTH, add_special_tokens=True)
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

        for i, (text, label) in enumerate(processedData['test']):
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

        # Load the BERT embedding model
        bertModel = BertModel.from_pretrained(
            'bert-base-uncased', output_hidden_states=True)
        bertModel.eval()    # Only wants to use the bert model

        # Convert inputs to pytorch tensor
        tensorData = {
            'train': {
                'X': [],
                'y': [],
            },
            'test': {
                'X': [],
                'y': []
            }
        }
        for i, each in enumerate(inputData['train']['X']):
            tokenTensor = torch.tensor(each['input_ids']).unsqueeze(dim=0)
            # BERT is trained and expect sentence pairs, so we need to number each tensor to belong to a text
            segmentTensor = torch.tensor([i] * BERT_MAX_LENGTH).unsqueeze(dim=0)
            with torch.no_grad():
                output = bertModel(tokenTensor, segmentTensor)
                hidden_states = output[2]

            # Cut the layer to feed to RNN            
            # test = torch.stack(hidden_states, dim=0)[-1].squeeze()
            # print(f'The shape of tensor: {test.shape}')


            inputTensorToModel = torch.stack(
                hidden_states, dim=0)[-1].squeeze()
            tensorData['train']['X'].append(inputTensorToModel)
            tensorData['train']['y'].append(inputData['train']['y'][i])

        for i, each in enumerate(inputData['test']['X']):
            tokenTensor = torch.tensor(each['input_ids']).unsqueeze(dim=0)
            # BERT is trained and expect sentence pairs, so we need to number each tensor to belong to a text
            segmentTensor = torch.tensor([i] * BERT_MAX_LENGTH).unsqueeze(dim=0)
            with torch.no_grad():
                output = bertModel(tokenTensor, segmentTensor)
                hidden_states = output[2]
            # Cut the layer to feed to RNN
            inputTensorToModel = torch.stack(
                hidden_states, dim=0)[-1].squeeze()
            # tuple of (n=input tensor, segment tensor)
            tensorData['test']['X'].append(inputTensorToModel)
            tensorData['test']['y'].append(inputData['test']['y'][i])

        # Save the conversion into tensor
        if save:
            with open(f'P4/saved/{self.task}-dataInTensor', 'wb') as handle:
                pickle.dump(tensorData, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

        return tensorData

    def useSavedData(self) -> dict:
        with open(f'P4/saved/{self.task}-dataInTensor', 'rb') as handle:
            tensorData = pickle.load(handle)
        print(tensorData['train']['X'][1].shape)
        return tensorData


test = Dataset_Loader(task='test_data')
res = test.load(save=True)
# print(res['train']['X'])
print(res['train']['X'][1])
print(len(res['train']['X']))
print(res['train']['X'][1].shape)