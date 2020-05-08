import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pickle
from keras.models import model_from_json
import json


fp = open('datafile.pkl','rb')
data = pickle.load(fp)
fp.close()

chars = data['chars']
charlen = data['charlen']
maxlen = data['maxlen']

lcase_table = u'abcçdefgğhıijklmnoöprsştuüvyz'
ucase_table = u'ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ'

def upper(data):
    data = data.replace('i',u'İ')
    data = data.replace(u'ı',u'I')
    result = ''
    for char in data:
        try:
            char_index = lcase_table.index(char)
            ucase_char = ucase_table[char_index]
        except:
            ucase_char = char
        result += ucase_char
    return result

def lower(data):
    data = data.replace(u'İ',u'i')
    data = data.replace(u'I',u'ı')
    result = ''
    for char in data:
        try:
            char_index = ucase_table.index(char)
            lcase_char = lcase_table[char_index]
        except:
            lcase_char = char
        result += lcase_char
    return result

def capitalize(data):
    return data[0].upper() + data[1:].lower()

def title(data):
    return " ".join(map(lambda x: x.capitalize(), data.split()))


def encode(word,maxlen=22,is_pad_pre=False):
	wlen = len(word)
	if wlen > maxlen:
		word = word[:maxlen]
		
	word = lower(word)
	pad = maxlen - len(word)
	if is_pad_pre :
		word = pad*' '+word   
	else:
		word = word + pad*' '
	mat = []
	for w in word:
		vec = np.zeros((charlen))
		if w in chars:
			ix = chars.index(w)
			vec[ix] = 1
		mat.append(vec)
	return np.array(mat)   
 
def decode(mat):
	word = ""
	for i in range(mat.shape[0]):
		word += chars[np.argmax(mat[i,:])]
	return word.strip().split()[0]


def tokenize(input):
    if(input == None):
        return None
    jstr = json.loads(open('kokbul.json').read())
    model = model_from_json(jstr)
    model.load_weights('kokbul-18-0.98.hdf5')
    x = []

    w = encode(input)
    x.append(w)
    x = np.array(x)

    yp = model.predict(x)
    return decode(yp[0])


if __name__ == "__main__":
    print("------------------------------------------")
    print("Predicted : ",tokenize("bağışı"))
    print("------------------------------------------")