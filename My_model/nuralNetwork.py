import numpy as np
import random
from sigmoidNuronLayer import Layer

class nw(object):
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


    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a)+b)
        return a


    def sigmoid(self, z):
        """The sigmoid activation function."""
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self, z):
        """Derivative of the sigmoid function."""
        sig = self.sigmoid(z)
        return sig * (1 - sig)


    def SGD(self, trainingData, epochs, miniBatchSize, learningRate, testData=None, test_data=None):
        if test_data is None:
            test_data = testData
        if test_data:
            numTests = len(test_data) # if optional argument test_data is given then count the number of tests.

        n = len(trainingData)

        for epoch in range(epochs): # for each epoch
            random.shuffle(trainingData) #shuffle the training data

            #then find all possible minibatches of the shuffled data
            miniBatches = []
            for miniBatchNo in range(0, n, miniBatchSize): 
                miniBatch = trainingData[miniBatchNo : miniBatchNo+miniBatchSize]
                miniBatches.append(miniBatch)

            #and update their values
            for miniBatch in miniBatches:
                self.updateMiniBatch(miniBatch, learningRate)

            #now give stats if test data is given
            if test_data: 
                print(f"Epoch {epoch}, {self.evaluate(test_data)}, / {numTests}")


    def updateMiniBatch(self, miniBatch, learningRate):
         """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
         nabla_b = [np.zeros(b.shape) for b in self.biases]
         nabla_w = [np.zeros(w.shape) for w in self.weights]
         for x, y in miniBatch:
             delta_nabla_b, delta_nabla_w = self.backprop(x, y)
             nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
             nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
         self.weights = [w-(learningRate/len(miniBatch))*nw
                         for w, nw in zip(self.weights, nabla_w)]
         self.biases = [b-(learningRate/len(miniBatch))*nb
                        for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives of the cost with respect
        to the output activations."""
        return (output_activations-y)
    
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))
                