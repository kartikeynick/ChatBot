import json
from nltk_down import tokenize, stem, bagOfWords
import numpy as np

from NLPModel import NeuralNetwork

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# to load the json file
with open('intents.json', 'r') as f:
    intents = json.load(f)

# print(intents)

allWords = []  # empty list
tags = []  # for tags
xy = []  # later holds all words and tags

for intent in intents['intents']:
    t = intent['tag']  # key tag as in the json file
    tags.append(t)  # then we will be appending it in out tag array and
    for pattern in intent['patterns']:  # loop through every pattern with this tag in the json file
        tknize = tokenize(pattern)
        allWords.extend(tknize)  # append it into the all word array
        xy.append((tknize, t))  # so it will know the pattern and the corresponding tag

# Now excluding the punctuation characters, Later Implication: Removing all the stopwords later down the road if
# needed, but for now this works


ignoreWords = ['?', '!', '.', ',']  # Array of punctuation characters. We Will not be needing them for our BOW model
allWords = [stem(w) for w in allWords if w not in ignoreWords]  # do Stemming
# print(allWords)  # This will give a tokenized and stemmed word and removed the special characters

allWords = sorted((set(allWords)))  # sort and remove all the duplicate words
tags = sorted((set(tags)))  # removing and sort all the tag duplicates
# print(tags)

aTrain = []  # bag of words
bTrain = []  # associated number for each tags

# loop over our xy array
for (tknize, tag) in xy:
    # calling from the nltk_down so here tknize is already tokenized
    bag = bagOfWords(tknize, allWords)
    # then append it into train array
    aTrain.append(bag)
    # numbers for our labels as in 0,1,2,....n
    lable = tags.index(tag)
    bTrain.append(lable)  # CorssEntropyLoss that is why not doing 1 hot encoding

# after this we have to convert it into numpy array
aTrain = np.array(aTrain)
bTrain = np.array(bTrain)


# PyTorch model and training


class CDataset(Dataset):  # ChatDataset --> Dataset parameter from torch so it inherit the dataset
    def __int__(self):
        self.n_samples = len(aTrain)  # store number of samples a train array
        self.adata = aTrain
        self.bdata = bTrain

    def __getitem__(self, index):
        return self.adata[index], self.bdata[index]

    def __len__(self):
        return len(aTrain)  # return self number of sammples


# hyperparameters
batch_size = 8

dataset = CDataset()

inps = len(aTrain[0])  # input size --> len of each BOG we creater --> it has the same len as the all word array

hids = 8  # hidden size
outs = len(tags)  # output size --> number of diff tags we have
# print(inps,len(allWords))
# print(outs,tags)


# create a data loader here
# batch size=8 (lets)
# number of workers =2 --> it is for multiprocessing so that it works faster


train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)  # chatdataset


nlpModel= NeuralNetwork(inps,hids,outs)

datasave={
    "model_state": nlpModel.state_dict(),
    "output_size": inps,
    "output_size": outs,
    "hidden_size": hids,
    "all_words": allWords,
    "tags": tags
}

filef="data.pth"
torch.save(datasave,filef)

print(f' Completed the tried data and it is stored in {filef}')