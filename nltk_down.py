import nltk
#nltk.download('punkt') #pretrained tokenizer package
from nltk.stem.porter import PorterStemmer #for stemmer
import numpy as np
stemmer=PorterStemmer() #creater a stemmer
def tokenize (sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())# return all to the lowercase

def bagOfWords(tokenizedSentence,all_words): #tokenized sentence and all words
    '''S=['hello','how','are','you']
    words=['hi','hello','i','you','bye','thank','cool']

    bog= [0,1,0,1,0,0,0]
    making bag of words
    '''
    #Now by using the list comprehension.
    tokenizedSentence = [stem(w) for w in tokenizedSentence]
    print(tokenizedSentence)
    bag=np.zeros(len(all_words),dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenizedSentence:
            bag[idx]=1.0
    return bag

#S=['hello','how','are','you']

S=["organize", "organization","organizing"]
words=['hi','hello','i','you','bye','thank','cool']
bog=bagOfWords(S,words)
print(bog)


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