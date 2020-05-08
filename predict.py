import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utilizenn import loadData, loadModel, retrieveResponse, oneHotEncodingInput
import time

class Predict:
    def __init__(self):
        self.words, self.labels, self.dataorigin = loadData()
        self.model = loadModel(len(self.words),len(self.labels))
        #print(self.labels)

    def getResult(self, input, n):
        start = time.time()
        
        output = oneHotEncodingInput(input,self.words)
        output = self.model.predict([output])
        output = retrieveResponse(output,self.dataorigin, n)
        
        end = time.time()

        elapsepTime = end - start
        print("Resulted in -- " + str(elapsepTime) + " seconds")

        return output

#for testing purposes
if __name__ == '__main__':
    a = Predict()
    print(a.getResult('mesaj ile Nasıl Bağış Yaparım', 5))