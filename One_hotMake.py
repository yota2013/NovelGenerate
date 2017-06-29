#coding:utf-8

import numpy as np
import MeCab
import glob
import re
from keras.preprocessing.text import one_hot
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Embedding
from keras.optimizers import RMSprop
import math
import sys
import random

#np.set_printoptions(threshold=np.inf)


"""
#データを読み込む
#mecabで文章分割する．
#データをonehotに変換するプログラムを作成する．


"""

#class Corpus:
#        def __init__(self):

#def make_one_hot(text):

    #index = self.vocabulary_size - 1 if char == " " else (ord(char) - ord("a"))
    #value = np.zeros(self.vocabulary_size)
    #value[index] = 1
    #return value

#RNNで学習させる
#LSTM， ．
def ModelLstm(X,y,length,dictionary):
    maxlen = 3
    model = Sequential()
    model.add(LSTM(128, input_shape=(3, length))) #128batch_size
    model.add(Dense(length))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    def sample(preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    for iteration in range(1, 60):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(X, y,
                  batch_size=128,
                  epochs=1)
        start_index = random.randint(0, len(dictionary) - maxlen - 1)

        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            print('----- diversity:', diversity)

            generated = ''
            sentence = dictionary[start_index: start_index + maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)
            #前から３つずつ取っている．
            for i in range(400):
                x = np.zeros((1, maxlen, length))
                x[0] = x[0] + X[0,random.randint(0,1000)]
                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)
                #next_char = indices_char[next_index]

                #generated += next_char
                #sentence = sentence[1:] + next_char

                sys.stdout.write(next_index)
                sys.stdout.flush()
            print()


def MakeOnehot(texts,V,dictionary):
    print("-------------MakeOnehot-------------")
    f = open('text.txt', 'w')
    f.write(str(texts.split(" ")))
    f.close()

    sent_vec = one_hot(texts, V, lower=False, split=' ',filters='')
    print("sent_vec",len(sent_vec))
    oneHot = to_categorical(sent_vec)#onhotvectorの作成

    vectorAndWordList = []
    maxlen = 3
    sentencelen = []

    #onehovectorと単語の辞書を作成する
    id = 0
    lesslen = 20
    print("oneHot",len(oneHot[:,0]))
    """
    for numberSentence,sentence in enumerate(dictionary):
        for i,word in enumerate(sentence):
            WordList.append([word,oneHot[i+id]])
        sentencelen.append(i+1)
        if(lesslen > i+1):
            lesslen = i + 1
        id = id + i + 1
    print("vector",len(vectorAndWordList[0][1]))

    print(lesslen)
    if(len(vectorAndWordList[0][1]) != V):#修正
        V = len(vectorAndWordList[0][1])
    """
    #全部の文を全て3-gram
    x = []
    y = []
    for numberSentence,sentence in enumerate(dictionary):
        vectorAndWordList = []
        nextVector = []
        textlen = len(sentence)
        sentencelen.append(textlen)
        #for i,word in enumerate(sentence):
        for l in range(0,textlen - maxlen,1):
            vectorAndWordList.append(oneHot[l+id: l + id + maxlen])
            nextVector.append(oneHot[l+id+maxlen])
        id = id + textlen
        #vectorAndWordList = np.array(vectorAndWordList, dtype=object)
        if (len(vectorAndWordList[0][1]) != V):  # 修正
            V = len(vectorAndWordList[0][1])
        #print(len(vectorAndWordList),len(nextVector))
        wordHot = np.zeros((len(vectorAndWordList), maxlen, V), dtype=np.bool)  # wordHotする numbersentence がループが0からだから
        nextWord = np.zeros((len(vectorAndWordList), V), dtype=np.bool)  # nextWordを代入する
        #print(len(vectorAndWordList[1]),len(wordHot[1]))
        for i in range(0,textlen - maxlen,1):
            wordHot[i] = wordHot[i] + vectorAndWordList[i]
            nextWord[i] = nextWord[i] + nextVector[i]
        nextWord = np.array(nextWord)
        wordHot  = np.array(wordHot)
        print(nextWord.shape)
        print(wordHot.shape)
        x.extend(wordHot)
        y.extend(nextWord)

    #print("wordHot",len(wordHot[2,1]))
    #print(x)
    x = np.array(x)
    print(x.shape)
    y = np.array(y)
    print(y.shape)
    #print(x)
    #print(y)
    #for numberSentence,sentence in enumerate(dictionary):
        #print(sentence)
        #sentenceDatalen = math.floor(len(sentence)/4.0)
        #TempData = np.zeros((sentenceDatalen, maxlen, V), dtype=np.bool)
        #for i,word in enumerate(sentence):
        #    for l in range(0,len(sentence) - maxlen):
            #wordHot[sentenceDatalen,i:i+maxlen] = wordHot[numberSentence,i:i+maxlen] + vectorAndWordList[i,1]
            #nextdHot[numberSentence, i] = wordHot[numberSentence, i] + vectorAndWordList[i, 1]

    #f = open('text.txt', 'w')
    #f.write(str(wordHot[1]))


    """
    Onehotの確認のためのプログラムを作成した．
    """
    #print(VectorAndWord[0])
    #print(id,len(oneHot),len([flatten for inner in VectorAndWord for flatten in inner])/2)
    #f = open('text.txt', 'w')
    #for wordlist in VectorAndWord:
    #    if (wordlist[0] == "EOS"):
    #        f.write(str(wordlist[1]))
    #f.close()


    return x,V,y
    #onh_hot_dictionary = {}
    #for text_one,coupustext in zip(dictionary,coupurs):


def FileRead(dataDir):
    #データ読み込み
    texts = ""
    dictionary = []
    mt = MeCab.Tagger("-Oyomi")
    for filename in glob.glob("./"+dataDir+"/NovelTile201.txt"):
        with open(filename, "r", encoding="utf-8") as f:
            for n, text in enumerate(f.readlines()):
                #print(text)
                text = text.replace("\u3000", "")
                text = re.sub(re.compile(u"[!-/:-@[-`{-~]☆◈◯≧≪♀♂▼★♨◈〃"), "", text)
                text = text+"EOS"
                textpase = mt.parse(text)#mecab
                textpase = textpase.replace(" \n", "")
                textpase = textpase.replace("\n", "")
                textpase = textpase.replace(" ", "")
                textpase = list(textpase)
                #print(textpase)
                texts = texts+" ".join(textpase)+" "
                dictionary.append(textpase)
    #print(dictionary)
    #print(texts)
    #漢字と日本語をなくすことする
                #print(textpase)
    #print(dictionary)
    #V = len(set(dictionary)) + 1
    #print(V)
    flattendictionary = [flatten for inner in dictionary for flatten in inner]
    print("flattendictionary",len(flattendictionary))
    vocabrary=set(flattendictionary)
    print(vocabrary)
    V = len(vocabrary) + 1
    print(V)
    #print(V)

    return texts,V,dictionary










def main():
    texts,V,dictionary = FileRead("DATA")
    wordHot,length,nextWord= MakeOnehot(texts,V,dictionary)
    ModelLstm(wordHot,nextWord,length,dictionary)

if __name__ == '__main__':
    main()