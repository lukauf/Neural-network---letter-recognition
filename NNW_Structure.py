import numpy

class MLP:
        def __init__(self, n_input, n_hidden, n_output): #120 input neurons, 60 (arbitrary number) neurons at hidden layer, and 26 output neurons, (26 letters/results)

                self.W1 = numpy.random.randn(n_input, n_hidden) * 0.01 #weight matrix 120x60 input layer -> hidden layer
                self.W2 = numpy.random.randn(n_hidden, n_output)  * 0.01 #weight matrix hidden layer -> output layer
                self.b1 = numpy.random.randn(1, 1) #bias matrix 60x1 hidden layer bias matrix
                self.b2 = numpy.random.randn()

        #each collumn represents the weights associated to a neuron

        #each line represents the weights which each neuron, from the previous layer, transmits to the neurons of the next layer


        @staticmethod
        def relu(z): #activation function that returns 0 if the z < 0 and returns z if z > 0, z is the total input of the neuron, sum considering the weights + the bias
                return numpy.maximum(0, z)


        @staticmethod
        def sigmoid(z): #activation function that returns values between [0,1] that gets
                return 1/(1 + numpy.exp(-z))

        def fowardpass(self,X): #first input at the network == X(1x120)
                Z = numpy.dot(X, self.W1) + self.b1 #X(1x120) x W1(120x60) + b1 = Z(1x60)
                X2 = self.relu(Z)#X2(1X60) the function relu is applied to every value of the array, so we get an array which every value is the return of this function that was applied on each value of the Z array
                Z2 = numpy.dot(X2, self.W2)#X2(1x60) x W2(60x26) = Z2(1x26)
                X3 = self.sigmoid(Z2)#X3(1x26), X3 has the values between 0 and 1, we will later get the biggest value to see which letter is it (each slot of the array is a reference to a letter of the alfabet)
                return X3










