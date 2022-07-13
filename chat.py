import random
import json
import torch

from NLPModel import NeuralNetwork
from nltk_down import bagOfWords, tokenize, stem

# opening the json file here
with open('intents.json', 'r') as f:
    intents = json.load(f)

file = "data.pth"
data = torch.load(file)

inps = data["input_size"]
hids = data["hidden_size"]
outs = data["output_size"]
allWords = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

nlpModel = NeuralNetwork(inps, hids, outs)
nlpModel.load_state_dict(model_state)
nlpModel.eval()

botName = "Bob"
print("Hey my name is Bob!, type exit to quit")

while True:
    s= input('User : ') # the raw sentence
    if s=='exit':
        break
    s=tokenize(s)
    x=bagOfWords(s,allWords) # taking the tokenized sentence and all the words from trained dataset
    #then reshape this
    x=x.reshape(1,x.shape[0])# 1 row
    x=torch.from_numpy(x)

    output=nlpModel(x)

    print("s = ",s,"\n x = ",x," out = ",output)
    _,preciction = torch.max(output, dim=1)
    tag=tags[preciction.item()] #then find the tag corresponding to the predicted

    #now apply a Softmax here
    probability=torch.softmax(output,dim=1)
    p=probability[0][preciction.item()] #this will be the probability

    print(" probability : ",p)

    if p.item()>0.02:
        for i in intents['intents']:
            if tag== i["tags"]:
                rw=random.choice(i['responses']) # pick random replies from responses
                print(f"{botName}: {rw}") # print the random
    else:
        print(f"{botName}: Dfaq you are you saying? Say it again nigg!")
