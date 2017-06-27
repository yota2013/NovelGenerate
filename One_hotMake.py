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


np.set_printoptions(threshold=np.inf)


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
def ModelLstm(onehot_vector,length):
    model = Sequential()
    model.add(LSTM(128, input_shape=(35, length))) #128batch_size
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
        #model.fit(X, y,
        #          batch_size=128,
        #          epochs=1)



def MakeOnehot(texts,V,dictionary):
    print("-------------MakeOnehot-------------")
    sent_vec = one_hot(texts, V, lower=False, split=" ")
    # print(sent_vec)
    oneHot = to_categorical(sent_vec)#onhotvectorの作成
    #f = open('text.txt', 'w')
    #f.write(str(oneHot[0]))
    #f.close()
    vectorAndWordList = []
    #onehovectorと単語の辞書を作成する
    id = 0
    maxlen = 0
    print("oneHot",len(oneHot[0]))

    for numberSentence,sentence in enumerate(dictionary):
        for i,word in enumerate(sentence):
            vectorAndWordList.append([word,oneHot[i+id]])
            #print(vectorAndWord)
            #wordHot.append(oneHot[i+id])
        #if (maxlen < max(i,maxlen)):
        if(maxlen < i+1):
            #print(word)
            #print(sentence)
            maxlen = i + 1
        id = id + i + 1
    print("vector",len(vectorAndWordList[0][1]))#34

    #print(zero)
    if(len(vectorAndWordList[0][1]) != V):#修正
        V = len(vectorAndWordList[0][1])

    wordHot = np.zeros((numberSentence+1, maxlen, V), dtype=np.bool)#wordHotする numbersentence がループが0からだから
    nextWord = np.zeros((numberSentence+1, V), dtype=np.bool)#nextWordを代入する
    vectorAndWordList = np.array(vectorAndWordList, dtype=object)
    print("wordHot",len(wordHot[2,1]))
    for numberSentence,sentence in enumerate(dictionary):
        #print(sentence)
        for i,word in enumerate(sentence):
            wordHot[numberSentence,i] = wordHot[numberSentence,i] + vectorAndWordList[i,1]
            nextdHot[numberSentence, i] = wordHot[numberSentence, i] + vectorAndWordList[i, 1]
    f = open('text.txt', 'w')
    f.write(str(wordHot[1]))


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


    return oneHot,V,vectorAndWordList
    #onh_hot_dictionary = {}
    #for text_one,coupustext in zip(dictionary,coupurs):


def FileRead(dataDir):
    #データ読み込み
    texts = ""
    dictionary = []
    mt = MeCab.Tagger("-Owakati")
    for filename in glob.glob("./"+dataDir+"/NovelTile2??.txt"):
        with open(filename, "r", encoding="utf-8") as f:
            for n, text in enumerate(f.readlines()):
                #print(text)
                text = text.replace("\u3000", "")
                text = re.sub(re.compile("[!-/:-@[-`{-~]"), " ", text)
                text = text + " " + "EOS"
                textpase = mt.parse(text)#mecab
                textpase = textpase.replace(" \n", "")
                texts = texts + " " + textpase
                textpase = textpase.split(" ")
                dictionary.append(textpase)

                #print(textpase)
    #print(dictionary)
    #V = len(set(dictionary)) + 1
    #print(V)
    flattendictionary = [flatten for inner in dictionary for flatten in inner]
    vocabrary=set(flattendictionary)
    #print(vocabrary)
    V = len(vocabrary) + 1
    print(V)
    #print(V)
    return texts,V,dictionary










def main():
    texts,V,dictionary = FileRead("DATA")
    oneHot,length,vectorAndWordList= MakeOnehot(texts,V,dictionary)
    #ModelLstm(oneHot,length),

if __name__ == '__main__':
    main()