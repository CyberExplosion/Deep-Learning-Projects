# Custom made for BERT

from code.base_class.dataset import dataset
import pandas as pd
from transformers import BertTokenizer, BertModel
from pathlib import Path
import re
import pickle
import nltk
from tqdm import tqdm, trange
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from torchtext.vocab import build_vocab_from_iterator
import torch

class Dataset_Loader(dataset):
    data = {}
    def __init__(self, dName=None, dDescription=None, task='test_data', inputLenNeedForSeq=3):
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
        self.numWordInput = inputLenNeedForSeq

    def loadTokenizedData(self) -> dict:
        with open(f'{self.picklePath}/{self.task}-tokenizedData.pickle', 'rb') as handle:
            tensorData = pickle.load(handle)
        return tensorData

    def tokenize(self, save=False) -> dict:
        # * 1 is for positive, 0 is negative
        print('loading data....')
        if self.task != 'text_generation':
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

        else: # ! Text Generation
            processedData = []
            generationDataPath = Path(self.dataPath, 'data')
            textIn = pd.read_csv(generationDataPath)
            for each in textIn.loc[:, 'Joke']:
                processedData.append(each)

            # Clean data
            print('cleaning data ...')
            cleanedData = []
            nltk.download('popular')
            CLEANHTML = re.compile('<.*?>')
            stopWords = set(stopwords.words('english'))
            for joke in processedData:
                text = re.sub(CLEANHTML, '', joke)
                # split into white space
                wordList = nltk.word_tokenize(text)
                # remove symbol and stop words
                wordList = [word.lower() for word in wordList if word.isalpha()]
                wordList.append('<EOS>')
                cleanedData.append(wordList)

            print(f'cleandData: {cleanedData}')

            # Build vocabs and tokenized
            vocabs = build_vocab_from_iterator(cleanedData, specials=['<UNK>'])
            vocabs.set_default_index(vocabs['<UNK>'])
            tokenizedData = []
            for joke in cleanedData:
                tokenized = vocabs(joke)
                tokenizedData.append(tokenized)
            vocab_size = len(vocabs)
            
            # Sequence text to input and output, adjustable
            seqLen = self.numWordInput + 1
            sequences = []
            for joke in tqdm(cleanedData, desc='Sequence generation progress'):
                for i in range(seqLen, len(joke)):
                    # select sequence of token
                    seq = joke[i-seqLen:i]
                    # convert to a line
                    line = ' '.join(seq)
                    # store
                    sequences.append(line)

            # Tokenized the Sequences
            tokenizedSequences = []
            for each in sequences:
                wordList = nltk.word_tokenize(each)
                tokenized = vocabs(wordList)
                tokenizedSequences.append(tokenized)

            # Build input and output
            X = []
            y = []
            for each in tokenizedSequences:
                if len(each) == seqLen:
                    X.append(each[:-1])
                    # Make y one hot encode
                    y.append(each[-1])
                else:
                    print(f'The not fit tokenized: {each}')
                    sentence = []
                    for word in each:
                        sentence.append(vocabs.lookup_token(word))
                    print(f'It representation is: {sentence}')

            # * USE THIS AFTER LOADING  
            # ! This for one hot encoding
            # tensorY = torch.nn.functional.one_hot(torch.tensor(y), num_classes=vocab_size)
            tensorY = torch.tensor(y)
            tensorX = torch.tensor(X)
            inputData = {
                'X': tensorX,
                'y': tensorY
            }

        if save:
            print('saving the tokenized data...')
            with open(f'{self.picklePath}/{self.task}-tokenizedData.pickle', 'wb') as handle:
                pickle.dump(inputData, file=handle)
            # save the vocab tokenizer
            with open(f'{self.picklePath}/{self.task}-tokenizer.pickle', 'wb') as handle:
                pickle.dump(vocabs, file=handle)


        return inputData
    

# test = Dataset_Loader(task='text_generation')
# res = test.tokenize(save=True)
# print(len(res['X']))
# print(res['X'][4])
# 
# print('Using saved, should have some output:')
# loaded = test.loadTokenizedData()
# print(len(loaded['X']))
# print(len(loaded['X'][4]))

# print(len(res['train']['X']))
# print(res['train']['X'][4])
# print(res['train']['X'][4])
# 
# print('Using saved, should have same output:')
# loaded = test.loadTokenizedData()
# print(len(loaded['train']['X']))
# print(loaded['train']['X'][4])
# print(loaded['train']['X'][4])