import numpy as np
from sigmoidNuron import SigmoidNuron
class Layer(object):
    nextLayerNo = 0
    
    def __init__(self, biaseList, weightList):
        self.layerNo = Layer.nextLayerNo
        Layer.nextLayerNo += 1 # generates a new unique layer number for each new layer
        
        self.biaseList = biaseList # ieth element of this list is a list of biases for ieth layer
        self.weightList = weightList
        # the ieth elemnt is a list X. 
        # weightList = [X1, X2, X3, ...]
        # And X is a list of Y elements. So X = [Y1, Y2, Y3, ...]
        # Y is a lis
        # Y = (weight for n1, weight for n2, ...)
        self.numNurons = self.biaseList.shape[0]
        self.layerInfo()

    def sigmoid(z):
        return(1.0/(1.0 + np.exp(z)))
    
    def layerInfo(self):
        print("#######>Layer Info:")
        print(f"#######>LayerNo: {self.layerNo+1} has {self.numNurons} nurons")

        for nuronNo in range(self.numNurons):
            nuron = SigmoidNuron(self.biaseList[nuronNo], self.weightList[nuronNo])
            print(f"#######>nuronID: {nuron.nuronId}, biase: {nuron.biase}, weights: {nuron.weights}")
        print("\n\n")

    

    def runNuronLayer(self, input):
        outputVector = (self.weightList * input) + self.biaseList