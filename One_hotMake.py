#coding:utf-8

import numpy as np
import MeCab
import glob
import re
from keras.preprocessing.text import one_hot
from keras.utils.np_utils import to_categorical
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


def MakeOnehot(texts,V):
    print("-------------MakeOnehot-------------")
    sent_vec = one_hot(texts, V, lower=False, split=" ")
    # print(sent_vec)
    oneHot = to_categorical(sent_vec)
    f = open('text.txt', 'w')
    f.write(str(oneHot[0]))
    f.close()
    #onh_hot_dictionary = {}
    #for text_one,coupustext in zip(dictionary,coupurs):


def FileRead(dataDir):
    #データ読み込み
    texts = ""
    dictionary = []
    mt = MeCab.Tagger("-Owakati")
    for filename in glob.glob("./"+dataDir+"/NovelTile???.txt"):
        with open(filename, "r", encoding="utf-8") as f:
            for n, text in enumerate(f.readlines()):
                #print(text)
                text = text.replace("\u3000", "")
                text = re.sub(re.compile("[!-/:-@[-`{-~]"), " ", text)
                textpase = mt.parse(text)
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
    return texts,V










def main():
    texts, V = FileRead("DATA")
    MakeOnehot(texts,V)

if __name__ == '__main__':
    main()