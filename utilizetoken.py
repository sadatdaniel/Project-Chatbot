import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pickle
from tensorflow.keras.models import model_from_json
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
    decoded = decode(yp[0])
    
    if similarity(decoded,"bağış") > 0.9:
        return "bağış"
    
    return decoded


if __name__ == "__main__":
    print("------------------------------------------")
    print("Predicted : ",tokenize("bagislarimiz"))
    print("------------------------------------------")