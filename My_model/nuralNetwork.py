import numpy as np
from sigmoidNuronLayer import Layer

class network(object):
    def __init__(self, nuronPerLayer):
        self.num_layers = len(nuronPerLayer)
        self.nuronPerLayer = nuronPerLayer
        
        self.biases = []
        self.initializeBiases()
        # the ieth element of biases list will contain a matrix representing the biases for all the nurons in the ieth layer
        
        self.weights = []
        self.initializeWeights()
        # now each element of weights is a list of weights for the i+1 eth layer of nurons
        # i starts from 0 in python

        self.initializeLayers()
        

    def getWeightCombinaitons(self, nuronPerLayer): #eg: [i1, h1, h2, o1] where i:input, h:hidden, o:output.
        inputAndHiddenLayer = nuronPerLayer[:-1] # [i1, h1, h2] we choose our firstLayer from this list, and ommit choosing the last layer, because that holds the output and dosent need a weight
        hiddenAndOutputLayer = nuronPerLayer[1:] # [h1, h2, o1] we choose our secondLayer from this list, and ommit the first layer because it is the input layer so no weights are applied to it

        #now making pares between neurons of two consecutive layers
        pares = []
        for i in range(0, len(nuronPerLayer)-1):
            pare = (inputAndHiddenLayer[i], hiddenAndOutputLayer[i])
            pares.append(pare) 
            # [ (i1,h1) , (h1,h2) , (h2, o1) ] these pares are (size of first layer, size of second layer) 
            # eg (2,3) means 2 nurons in first layer connect to 3 nurons in second layer
        return(pares)
    

    def initializeBiases(self):
        print("------------------------------------------------------------------------------------------------\nFUNCTION: INITIALIZING BIASES\n\n")
        for y in self.nuronPerLayer[1:]:# y = number of nurons in current layer
            biaseInLayer = np.random.randn(y,1) # make a yx1 dimention random initialized matrix. so y rows and 1 column. So the yeth row represents the biase for the yeth nuron in the layer
            self.biases.append(biaseInLayer) # one biase per neuron in the layer with y neurons.
            print(f"(-> The layer with {y} neurons has corresponding biases:\n\n{biaseInLayer}\n\n")
        print("DONE INITIALIZING BIASES\n------------------------------------------------------------------------------------------------")
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n\n")


    def initializeWeights(self):
        print("------------------------------------------------------------------------------------------------\nFUNCTION: INITIALIZING WEIGHTS\n")
        layersToConnect = self.getWeightCombinaitons(self.nuronPerLayer)

        # NOW FOR EACH of these pares, we make random weight matrix:
            #. each row represents the nuron of the secondLayer
            #. each column represents the nuron from the firstLayer
            #. the (x, y) pare's value is the weigth for the nuron y when it recieves input from nuron x

        #eg: (2,3) so we will make a matrix that will set the weights for the secondLayer which has 3 nurons.
        #So we will have a random matrix with 3 rows(3 nurons in second layer), and 2 columns(2 nurons in firstLayer)

            #  weight    i1nuron1  i1nuron2 
            #  h1nuron1  0.98      0.32
            #  h1nuron2  0.41      0.51
            #  h1nuron3  0.67      0.69

        # Here 0.98 is the weight when h1nuron1(secondLayer) recieves input from i1nuron1(firstLayer)

        for numFirstLayerNurons, numSecondLayerNurons in layersToConnect:
            secondLayerWeightMatrix = np.random.randn(numSecondLayerNurons, numFirstLayerNurons) # creates the weight matrix for a single layer
            print(f"(-> The weights for the layer with {numSecondLayerNurons} nurons are :\n\n{secondLayerWeightMatrix}\n\n")
            #Eg: if secondLayer has 3 nurons, and firstLauer has 2 nurons, will make a 3x2 matrix.
            self.weights.append(secondLayerWeightMatrix) # now each element of weights is a list of weights for the i+1 eth layer of nurons. i starts from 0 in python

        print("DONE INITIALIZING WEIGHTS\n------------------------------------------------------------------------------------------------")
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n\n")
    
    
    def initializeLayers(self):
        print("------------------------------------------------------------------------------------------------\nFUNCTION: INITIALIZING LAYERS\n")

        
        # now initializing one nuron layer OBJECT which will then initialize each nuron OBJECT.
        # the random values for the entire network have already been made, so now by making these objects we only make further iterations easier.
        for layerNo in range(1,self.num_layers): # since layer 0 has no weights or biases we start with layer 1

            numNuronsInLayer = self.nuronPerLayer[layerNo]
            print(f"(-> LayerNo: {layerNo} which will have {numNuronsInLayer} nurons as follows:\n")
            print(f"###>biases list:\n{self.biases[layerNo-1]}\n\n###>weights List:\n{self.weights[layerNo-1]}\n")
            layer = Layer(self.biases[layerNo-1], self.weights[layerNo-1])

        print("DONE INITIALIZING LAYERS\n------------------------------------------------------------------------------------------------")
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n\n")