from utilizenn import loadData, loadModel, retrieveResponse, oneHotEncodingInput
class Predict:
    def __init__(self):
        self.words, self.labels, self.dataorigin = loadData()
        self.model = loadModel(len(self.words),len(self.labels))

    def getResult(self, input):
        output = oneHotEncodingInput(input,self.words)
        output = self.model.predict([output])
        output = retrieveResponse(output,self.dataorigin)
        return output

#for testing purposes
if __name__ == '__main__':
    a = Predict()
    print(a.getResult('EĞİTİM İÇİN ÜNİVERSİTEYE DAVET'))