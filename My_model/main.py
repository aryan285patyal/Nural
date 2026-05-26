import nuralNetwork 
from sigmoidNuronLayer import Layer
import mnist_loader
import nuralNetwork

def main():
    # making a network with 2 input 3 hidden and 1 output
    # nn = network([2,3,1])
    training_data, validation_data, test_data = \
    mnist_loader.load_data_wrapper()
    net = nuralNetwork.nw([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, testData=test_data)


if __name__ == "__main__":
    main()

'''
run:
import mnist_loader
training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper()
import nuralNetwork
net = nuralNetwork.nw([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, testData=test_data)
'''