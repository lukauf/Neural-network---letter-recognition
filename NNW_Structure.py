import numpy

def init_parameters(n_input = 120, n_hidden = 60, n_output = 26): #120 input neurons, 60 (arbitrary number) neurons at hidden layer, and 26 output neurons, (26 letters/results)
        W1 = numpy.random.randn(n_hidden, n_input) #weight matrix 60x120
        b1 = numpy.random.randn(60, 1) #bias matrix 60x1
        return W1, b1


