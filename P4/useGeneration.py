from Method_Generation import Method_Generation
import torch
from pathlib import Path
import torchtext.vocab
import pickle
import nltk
import re
from nltk.corpus import stopwords

savedPath = Path('P4', 'result')
savedModel = '_generation_sLossFunction:CrossEntropy_sOptimizer:ADAM_sNumWordInput:3_sNumSequence:18585_sHiddenSize:50_sGradClipAmt:0.5_sDropout:0.5_sVocabSize:4368_sLearningRate:0.001_sMomentum:0.9_sMaxEpoch:500_sRandSeed:47.pt'
picklePath = Path('P4', 'saved')
task = 'text_generation'

model = Method_Generation()
model.load_state_dict(torch.load(Path(savedPath, savedModel)))

# Successfully load, now get each word, tokenize and tensorized

# Get tokenizer
with open(f'{picklePath}/{task}-tokenizer.pickle', 'rb') as handle:
    vocabs = pickle.load(file=handle)

# print(f'Print all vocab value\n {vocabs.get_itos()}')
# 
# print(f"The unknown vocab tokenized value: {vocabs.lookup_indices(['<UNK>'])}")

inputText = input()

# Clean data
# nltk.download('popular')
CLEANHTML = re.compile('<.*?>')
stopWords = set(stopwords.words('english'))

text = re.sub(CLEANHTML, '', inputText)
# split into white space
wordList = nltk.word_tokenize(text)
# remove symbol and stop words
cleanedInput = [word.lower() for word in wordList if word.isalpha()]

# cleanedInput.append('<UNK>')
# print(f'the cleaned input: {cleanedInput}')
# print(f'the tokenized value: {vocabs(cleanedInput)}')

# Make the input conform to 3 words only
while len(cleanedInput) > 3:
    cleanedInput.pop()
while len(cleanedInput) < 3:
    cleanedInput.append('<UNK>')

tokenizedInput = vocabs(cleanedInput)

# Pass through the model
tensorInput = torch.tensor(tokenizedInput).unsqueeze(dim=0)
res = model.test(tensorInput)[0]
nextWord = vocabs.lookup_token(res)

totalOutput = inputText + ' ' + nextWord
count = 20
while nextWord != '<EOS>' and count > 0:
    # new 3 words
    print(cleanedInput)
    cleanedInput.pop(0)
    cleanedInput.append(nextWord)
    # print(f'Value of new inputs: {newInput}')

    # tokenized again
    tokenizedInput = vocabs(cleanedInput)
    # Pass through model
    tensorInput = torch.tensor(tokenizedInput).unsqueeze(dim=0)
    res = model.test(tensorInput)[0]
    nextWord = vocabs.lookup_token(res)
    totalOutput += f' {nextWord}'
    count -= 1

print(totalOutput)