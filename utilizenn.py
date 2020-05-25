import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import string
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utilizetoken import tokenize
import pickle

def preprocessAword(s):
    #delete punctuation
    s = s.translate(str.maketrans('', '', string.punctuation))
    #delete whitespace
    s = s.translate(str.maketrans('', '', string.whitespace))
    #lowercase all
    s = preLower(s)
    s = s.lower()
    #find the root
    #print("preTokenize"+"--->"+s)
    s = tokenize(s)
    #print("postTokenize"+"--->"+s)
    #replace Turkish characters with English ones
    #tr = "çğıöşü"
    #eng = "cgiosu"
    #s = s.translate(str.maketrans(tr, eng))
    return s


def preLower(s):
    trU = "ÇĞIÖŞÜ"
    trL = "çğıöşü"
    s = s.translate(str.maketrans(trU, trL))
    return s

def loadData():
    try:
        with open("data.pickle", "rb") as f:
            words, labels, data_origin = pickle.load(f)
            return words, labels, data_origin
    except Exception as error:
        print("ERROR loadModel() --> " + repr(error))

def oneHotEncodingInput(input,words):
    bag = []
    wrds = [preprocessAword(s) for s in input.split()]
    for w in words:
        flag = False
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    
    if len(words) != np.sum(np.array(bag)):
        for elt in wrds:
            if elt not in words:
                index = findSimilarIndex(elt,words)
                bag[index] = 1            

    return bag

def findSimilarIndex(s, words):
    biggest = 0
    biggestIndex = 0
    for i in range(len(words)):
        similarityPoint = similarity(s,words[i])
        if(similarityPoint >= biggest):
            biggest = similarityPoint
            biggestIndex = i
            if(words[i] == "bağış"):
                break
    print(s + "-->"+ words[biggestIndex] + "-->" + str(biggestIndex))
    #print(words)
    return biggestIndex

def getOrderedResult(output):
    results = np.sort(output[0])
    results = results[::-1]
    orderedResults = []
    for i in range(len(results)):
        seen = np.where(output[0] == results[i])
        orderedResults.append(seen[0][0])
    return orderedResults

def retrieveResponse(output,data_origin,n):
    response_ids = getOrderedResult(output)
    response_dict = {}

    if(n > len(response_ids)):
        n = len(response_ids)
    
    for i in range(n):
        for index, row in data_origin.iterrows():
            if(row['RESPONSE_ID'] == response_ids[i]):
                possibility = output[0][response_ids[i]] * 100
                possibility = int(possibility)
                response_dict[possibility] = row['RESPONSE']
                break
    
    response_list = list(response_dict)
    if response_list[0] < 50:
        print("Possiblity --> " + str(response_list[0]))
        return {0 : "Please try again!!!!"}
    
    
    return response_dict


def getModel(words_length,labels_length):
    model = keras.Sequential([
    layers.Dense(16, activation='relu', input_dim=words_length),
    layers.Dense(8, activation='relu'),
    layers.Dense(labels_length, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(0.001)

    model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
    return model

def loadModel(words_length,labels_length):
    model = getModel(words_length,labels_length)
    checkpoint_path = "training/cp.ckpt"
    model.load_weights(checkpoint_path)
    return model

def similarity(s1,s2):
    tr="ığüşiöçIĞÜŞİÖÇ"
    en='igusiocigusioc'
    ss1=""
    ss2=""
    for i in range(len(s1)):
        contain=False
        for j in range(len(tr)):
            if s1[i]==tr[j]:
                ss1=ss1+en[j]
                contain=True
                continue
        if not contain:
            ss1=ss1+s1[i]
    ss1=str.lower(ss1)


    for i in range(len(s2)):
        contain=False
        for j in range(len(tr)):
            if s2[i]==tr[j]:
                ss2=ss2+en[j]
                contain=True
                continue
        if not contain:
            ss2=ss2+s2[i]
    ss1=str.lower(ss1)
    ss2=str.lower(ss2)

    list1=ss1.split(' ')
    list2=ss2.split(' ')
    sameChar = 0
    mostSimilar=0
    total=0
    for j in list1:
        for k in list2:
            sameChar=0
            for i in range(min(len(j),len(k))):
                if j[i]==k[i]:
                    sameChar+=1
            similar=sameChar/len(j)
            p1=0
            p2=0
            while p1<len(j) and p2<len(k):
                if j[p1]==k[p2]:
                    p1+=1
                p2+=1
            abb=0.8*p1/len(j)
            mostSimilar=max(mostSimilar,similar,abb)
        total*=1.5
        total+=mostSimilar
        mostSimilar=0
    return total/len(list1)

#for testing purposes
if __name__ == '__main__':
    #pass
    print(similarity("bagi","bağış"))
    #print(oneHotEncodingInput("reeewr BAĞISLADIM",["reeewr","bağış","şube","atlat","wet","bar","bagı"]))