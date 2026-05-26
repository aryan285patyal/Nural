import numpy as np

class SigmoidNuron(object):
    nextNuronId = 0
    def __init__(self, biase, weightsList):
        self.nuronId = SigmoidNuron.nextNuronId
        SigmoidNuron.nextNuronId += 1

        self.biase = biase
        self.num_inputs = len(weightsList) # first dimention of the np array, no of columns
        #self.input_connections = inputConnectionsList
        # the nuronIds to which currnt nuron is connected on the input side
        # nuronIds[0] will give the nuronId for the weight weightsList[0]
        # so ieth element of the nuronIds and weightsList correspond to the properties of the same nuron
    
        
        #self.outputNuronId = outputNuronId

        self.weights = weightsList # the weights for each input
        self.input = [] # has the input when running the nuron
        

    #def makeWeightInputAssignment(self):
        


    # def printNuron(self):
    #     print(f"nuron no. {self.nuronId} has biase {self.biase} and weights:\n")
    #     for weight in self.weights:
    #         print("from nuronID:{} weight is: {}\n")

    # def nuronInstance(self, fromNuron, input):
    #     #fromNuron : required to chose which weight to use
    #     #input : the actial value sent from the nuron before
