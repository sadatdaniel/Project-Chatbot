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

def retrieveResponse(output,data_origin):
    response_id = np.argmax(output[0])
    #print(response_id)
    
    for index, row in data_origin.iterrows():
        if(row['RESPONSE_ID'] == response_id):
            return row['RESPONSE']


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
    pass
    #print(preprocessAword('tRıaL ğğ öö? .,;:!s'))
    #words, labels, data_origin = loadData()
    #print(data_origin.head())
    #output =oneHotEncodingInput('Bağışımı x istiyorum',['x','y','z','m','n','h'])
    #print(output)