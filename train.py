import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utilizenn import preprocessAword, getModel, oneHotEncodingInput, retrieveResponse
import pickle

#read excel file into a dataframe
my_sheet = 'Sayfa1'
file_name = "Arranged_WebChat_KonuBasliklari.xlsx"
data_origin = pd.read_excel(file_name, sheet_name = my_sheet)

#drop unneccessary columns --> web chat, dil
data_origin.drop([data_origin.columns[1]],axis=1,inplace=True)
data_origin.drop([data_origin.columns[1]],axis=1,inplace=True)

#drop rows that has a null value in it
#data_origin.dropna(inplace=True)

#resetting index after drop (an unneccessary step :D)
#data_origin.reset_index(drop=True)

words = []
labels = []
docs_x = []
docs_y = []


for index, row in data_origin.iterrows():
    wrds = [preprocessAword(s) for s in row['ENTRY'].split()]
    words.extend(wrds)
    
    docs_x.append(wrds)
    docs_y.append(row['RESPONSE_ID'])

    labels.append(row['RESPONSE_ID'])

words = sorted(list(set(words)))
words = list(filter(None, words))

labels = sorted(list(set(labels)))

X = []
Y = []

out_empty = [0 for _ in range(len(labels))]

for i in range(len(docs_x)):
    bag = []

    for w in words:
        if w in docs_x[i]:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[i])] = 1
    
    X.append(bag)
    Y.append(output_row)

X = np.array(X)
Y = np.array(Y)
#print(X.shape)
#print(Y.shape)


#creating the network
model = getModel(len(words),len(labels))
EPOCHS = 350
# Create a callback that saves the model's weights
checkpoint_path = "training/cp.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                save_weights_only=True,
                                                verbose=0
                                                )

#training the model
model.fit(X,Y,epochs=EPOCHS, callbacks=[cp_callback], verbose=1)

#A painful debugging happened here
#output = oneHotEncodingInput('ODUL/MADALYA SISTEMI NEDIR?',words)
#print(output)
#output = model.predict([output])
#print(output)
#output = retrieveResponse(output,data_origin)
#print(output)

#saving useful information
with open("data.pickle", "wb") as f:
   pickle.dump((words, labels, data_origin), f)