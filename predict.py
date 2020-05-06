from utilizenn import loadData, loadModel, retrieveResponse, oneHotEncodingInput
import time

class Predict:
    def __init__(self):
        self.words, self.labels, self.dataorigin = loadData()
        self.model = loadModel(len(self.words),len(self.labels))

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
    print(a.getResult('İLK YARDIM MERKEZİ VE DETAYI', 3))