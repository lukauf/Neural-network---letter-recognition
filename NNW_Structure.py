import numpy

def init_parameters(n_input = 120, n_hidden = 60, n_output = 26): #120 input neurons, 60 (arbitrary number) neurons at hidden layer, and 26 output neurons, (26 letters/results)

        W1 = numpy.random.randn(n_input, n_hidden) * 0.01 #weight matrix 120x60 input layer -> hidden layer
        W2 = numpy.random.randn(n_hidden, n_output)  * 0.01 #weight matrix hidden layer -> output layer
        b1 = numpy.random.randn(1, 1) #bias matrix 60x1 hidden layer bias matrix
        b2 = numpy.random.randn()
        return W1, b1, W2, b2
#each collumn represents the weights associated to a neuron

#each line represents the weights which each neuron, from the previous layer, transmits to the neurons of the next layer



def relu(z): #activation function that returns 0 if the z < 0 and returns z if z > 0, z is the total input of the neuron, sum considering the weights + the bias
        return numpy.maximum(0, z)

def sigmoid(z): #activation function that returns values between [0,1] that gets
        return 1/(1 + numpy.exp(-z))

