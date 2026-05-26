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

        self.weights = weightsList # the weights for each input. so for n1 of the layer w1 is at index weightsList[1]
        
        self.input = [] # has the input when running the nuron
        self.printNuron()

    #def makeWeightInputAssignment(self):
        

    def sigmoidFunction(x):
        return 1.0/(1.0+np.exp(-x))


    def printNuron(self):
        print(f"\n###########>nuronId {self.nuronId} has biase {self.biase} and weights:")
        for i in range(len(self.weights)):
            print(f"###########>from nuronNo:{i} in previous layer, weight is: {self.weights[i]}")
        print("")

    def runNuron(self,input):
        #fromNuron : required to chose which weight to use
        #input : the actial value sent from the nuron before
        if np.shape(input) != np.shape(self.weights):
            print("NURON ERR: The shape of input and nurons's weights dosent match")
            return(0)
        
        print(input.shape[0])
        return (self.sigmoidFunction((self.weights*input) + self.biase))