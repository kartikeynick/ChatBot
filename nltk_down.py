import nltk
from nltk.stem.porter import PorterStemmer  # for stemmer
import numpy as np


stemmer = PorterStemmer()  # create a stemmer
def stem(w):
    return stemmer.stem(w.lower())  # return all to the lowercase with the root words of each words passes in the parameter

def tokenize(s):
    #it will tokenize the given sentense
    return nltk.word_tokenize(s)


def bagOfWords(ts, all_words):  # tokenized sentence and all words
    '''S=['hello','how','are','you']
    words=['hi','hello','i','you','bye','thank','cool']

    bog= [0,1,0,1,0,0,0]
    making bag of words
    '''
    # Now by using the list comprehension.
    tokenizedSentence = [stem(w) for w in ts] # do stemming for all the words in the tokenized sentence
    #print(tokenizedSentence)
    bag = np.zeros(len(all_words), dtype=np.float32)
    for i, w in enumerate(all_words):
        if w in tokenizedSentence:
            bag[i] = 1
    return bag


'''S=['hello','how','are','you']

#S=["organize", "organization","organizing"]
words=['hi','hello','i','you','bye','thank','cool']
bog=bagOfWords(S,words)
print(bog)
'''

'''
#test the tokenizer
a="Heyo whatsup?"
print(a)
a=tokenize(a)
print(a)

#test the stemmer
words=["organize", "organization","organizing"]
s1=[stem(w) for w in words]
print(s1)
'''
