import string
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle

def preprocessAword(s):
    #delete whitespace
    s = s.translate(str.maketrans('', '', string.whitespace))
    #delete punctuation
    s = s.translate(str.maketrans('', '', string.punctuation))
    #lowercase all
    s = s.lower()
    #replace Turkish characters with English ones
    tr = "çğıöşü"
    eng = "cgiosu"
    s = s.translate(str.maketrans(tr, eng))
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
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    return bag


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
                response_dict[str(i)] = row['RESPONSE']
                break
    return response_dict


def getModel(words_length,labels_length):
    model = keras.Sequential([
    layers.Dense(16, activation='relu', input_dim=words_length),
    layers.Dense(8, activation='relu'),
    layers.Dense(labels_length, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
    return model

def loadModel(words_length,labels_length):
    model = getModel(words_length,labels_length)
    checkpoint_path = "training/cp.ckpt"
    model.load_weights(checkpoint_path)
    return model


#for testing purposes
if __name__ == '__main__':
    #pass
    arr = np.array([[5,10,2,1,99]])
    print(getOrderedResult(arr))