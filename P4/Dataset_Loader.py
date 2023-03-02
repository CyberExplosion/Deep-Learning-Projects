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

    def cleanTrainData(self) -> list:
        cleanedData = []
        CLEANHTML = re.compile('<.*?>')

        for entry in self.processingData['train']:
            # ! Hope the data doesn't contain heavy html tags or else it wouldn't work
            text = re.sub(CLEANHTML, '', entry[0])
            cleanedData.append((text, entry[1]))
        return cleanedData

    # Warning this would not work for BERT model - Only for RNN
    def encodeForRNN(self) -> dict:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        BERT_MAX_LENGTH = 512

        for i, (text, label) in enumerate(self.processingData['train']):
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
            self.inputData['train']['X'].append(tokenized)
            self.inputData['train']['y'].append(label)

        for i, (text, label) in enumerate(self.processingData['test']):
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
            self.inputData['test']['X'].append(tokenized)
            self.inputData['test']['y'].append(label)

        return self.inputData

    def load(self, save=False) -> dict:
        # * 1 is for positive, 0 is negative
        print('loading data....')
        processedData = {
            'train': [],
            'test': []
        }
        dataPath = 'data/test_data'

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
                # keep special start and end symbol
                'input_ids': tokenized[0:1] + tokenized[-(BERT_MAX_LENGTH-1):],
                'token_type_ids': tokenType[0:1] + tokenType[-(BERT_MAX_LENGTH-1):],
                'attention_mask': attention[0:1] + attention[-(BERT_MAX_LENGTH-1):],
            }
            inputData['test']['X'].append(truncated)
            inputData['test']['y'].append(label)

            # Load the BERT embedding model
            bertModel = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
            bertModel.eval()    # Only wants to use the bert model

            # Convert inputs to pytorch tensor
            tokens_tensorList = []
            segments_tensorList = []
            for i, each in enumerate(inputData['train']['X']):
                tokenTensor = torch.tensor(each['input_ids'])
                # BERT is trained and expect sentence pairs, so we need to number each tensor to belong to a text
                segmentTensor = torch.tensor([i] * BERT_MAX_LENGTH)
                output = bertModel(tokenTensor, segmentTensor)
                hidden_states = output[2]
                # Cut the layer to feed to RNN
                inputTensorToModel = torch.stack(hidden_states, dim=0)[-1].squeeze()

                segments_tensorList.append(segmentTensor)
                tokens_tensorList.append(inputTensorToModel)


            # ! IN Data loader you load 1 by 1 => batch size of 1, make it work

        # Save the conversion
        if save:
            with open(f'P4/saved/{self.task}-dataInTensor', 'wb') as handle:
                pickle.dump(self.inputData, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

        return self.inputData

    def useSavedData(self) -> dict:
        with open(f'P4/saved/{self.task}-dataInTensor', 'rb') as handle:
            self.inputData = pickle.load(handle)
        return self.inputData

test = Dataset_Loader(task='test_data')
res = test.load(save=True)
print(res['train']['X'][0])
print(len(res['train']['X']))
