import json
from nltk_down import tokenize, stem,bagOfWords
import numpy as np

import torch
import  torch torch.nn as nn
from torch.utils.data import Dataset, DataLoader




#to load the json file
with open('intents.json', 'r') as f:
    intents=json.load(f)

#print(i)

allWords=[]# empty list
tags=[]# for tags
xy=[]#later holds all words and tags

for intent in intents ['intents']:
    tag=intent['tag'] #key tag as in the json file
    tags.append(tag) #then we will be appending it in out tag array and
    for pattern in intent['patterns']: # loop through every pattern with this tag in the json file
        tknize=tokenize(pattern)
        allWords.extend(tknize) #append it into the all word array
        xy.append((tknize,tag)) # so it will know the pattern and the corrosponding tag

#now excluding the punchuation characters, kind of removing all the stopwords

ignoreWords=['?','!','.',',']
allWords=[stem(w) for w in allWords if w not in ignoreWords]
print(allWords) #this will give a tokenized and stemmed word and removed the special characters
allWords=sorted((set(allWords))) # sort and remove all the duplicate words
tags=sorted((set(tags)))
print(tags)


aTrain=[]#bag of words
bTrain=[]#associated number for each tags

#loop over our xy array
for (tknize, tag) in xy:
    #caliing from the nltk_down so here tknize is already tokenized
    bag=bagOfWords(tknize, allWords)
    #then append it into atrain array
    aTrain.append(bag)
    #numbers for our lables as in 0,1,2,....n
    lable=tag.index(tag)
    bTrain.append(lable)#CorssEntropyLoss that is why not doing 1 hot encoding

#after this we have to convert it into numpy array
aTrain=np.array(aTrain)
bTrain=np.array(bTrain)

class CDataset(Dataset): #ChatDataset
    def __int__(self):
        self.n_samples=len(aTrain)
        self.adata=aTrain
        self.bdata=bTrain

        #dataset[idx]
        def __getitem__(self,index):
            return self.adata[idx],self.bdata[idx]
