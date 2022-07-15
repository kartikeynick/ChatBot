import json
from nltk_down import tokenize, stem, bagOfWords
import numpy as np
from NLPModel import NeuralNetwork
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# to load the json file
#with open('intents.json', 'r') as f:
with open('newIntent.json', 'r') as f:
    intents = json.load(f)

# print(intents)
# creating all the arrays to store the respective things

allWords = []  # empty list
tags = []  # for tags
xy = []  # later holds all words and tags

for i in intents['intents']:
    t = i['tag']  # key tag as in the json file
    tags.append(t)  # then we will be appending it in out tag array and
    for p in i['patterns']:  # loop through every pattern with this tag in the json file
        tknize = tokenize(p) #tokenize the respective patterns
        allWords.extend(tknize)  # append it into the all word array
        xy.append((tknize, t))  # so it will know the pattern and the corresponding tag

# Now excluding the punctuation characters, Later Implication: Removing all the stopwords later down the road if
# needed, but for now this works


ignoreWords = ['!','?', '.', ',']  # Array of punctuation characters. We Will not be needing them for our BOW model
allWords = [stem(w) for w in allWords if w not in ignoreWords]  # do Stemming and ignore all the ignore words
# print(allWords)  # This will give a tokenized and stemmed word and removed the special characters

allWords = sorted((set(allWords)))  # sort and remove all the duplicate words
tags = sorted((set(tags)))  # removing and sort all the tag duplicates
# print(tags)


# No here we will be creating training data

aTrain = []  # bag of words
bTrain = []  # associated number for each tags

# loop over our xy array
for (tknize, tag) in xy:
    # here a is bow for each pattern sentence hence tokenized
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
    def __init__(self):
        self.n_samples = len(aTrain) # store number of samples a train array
        self.AData = aTrain
        self.BData = bTrain

    # support the indexinf so the dataset[i] can be used to get the ith sample
    def __getitem__(self, i):
        return self.AData[i], self.BData[i]

    def __len__(self):
        return len(aTrain) # return self number of sammples


# hyperparameters
batch_size = 12
chatds = CDataset()
inps = len(aTrain[0])  # input size --> len of each BOG we creater --> it has the same len as the all word array
hids = 12  # hidden size
outs = len(tags)  # output size --> number of diff tags we have
# print(inps,len(allWords))
# print(outs,tags)
# create a data loader here
# batch size=8 (lets)
# number of workers =2 --> it is for multiprocessing so that it works faster


train_loader = DataLoader(dataset=chatds, batch_size=batch_size, shuffle=True, num_workers=0)  # chatdataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nlpModel= NeuralNetwork(inps,hids,outs).to(device)

# Loss and optimizer
lr=0.001#learning rat
num_epochs=1000

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(nlpModel.parameters(), lr=lr)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward the pass
        outputs = nlpModel(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

# Saving the data
datasave={
    "model_state": nlpModel.state_dict(),
    "input_size": inps,
    "output_size": outs,
    "hidden_size": hids,
    "all_words": allWords,
    "tags": tags
}

# riting the data into a file
filef="data.pth"
torch.save(datasave,filef)

print(f'\nCompleted the tried data and it is stored in {filef}')