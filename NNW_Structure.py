import numpy

class MLP:
        def __init__(self, n_input, n_hidden, n_output): #120 input neurons, 60 (arbitrary number) neurons at hidden layer, and 26 output neurons, (26 letters/results)

                self.W1 = numpy.random.randn(n_input, n_hidden) * 0.01 #weight matrix 120x60 input layer -> hidden layer
                self.W2 = numpy.random.randn(n_hidden, n_output)  * 0.01 #weight matrix hidden layer -> output layer
                self.b1 = numpy.random.randn(1, n_hidden) #bias matrix 1X60 hidden layer bias matrix
                self.b2 = numpy.random.randn(1, n_output)

        #each collumn represents the weights associated to a neuron

        #each line represents the weights which each neuron, from the previous layer, transmits to the neurons of the next layer


        @staticmethod
        def relu(z): #activation function that returns 0 if the z < 0 and returns z if z > 0, z is the total input of the neuron, sum considering the weights + the bias
                return numpy.maximum(0, z)


        @staticmethod
        def sigmoid(z): #activation function that returns values between [0,1] that gets
                return 1/(1 + numpy.exp(-z))

        @staticmethod
        def sigmoid_derivative(z):
                sig = 1 / (1 + numpy.exp(-z))
                return sig * (1 - sig)

        def forwardpass(self, X):
                self.Z = numpy.dot(X, self.W1) + self.b1 #X(Ntx120) x W1(120x60) + b1 = Z(Ntx60)
                self.X2 = self.relu(self.Z)#X2(NtX60) the function relu is applied to every value of the array, so we get an array which every value is the return of this function that was applied on each value of the Z array
                self.Z2 = numpy.dot(self.X2, self.W2)#X2(Ntx60) x W2(60x26) = Z2(1x26)
                self.X3 = self.sigmoid(self.Z2)#X3(Ntx26), X3 has the values between 0 and 1, we will later get the biggest value to see which letter is it (each slot of the array is a reference to a letter of the alfabet)
                return self.X3

        def BackPropagation(self,X, Y, alfa): #first input at the network == X(Ntx120) Nt = Number of examples offered as training samples, Y(Ntx26) = Expected output, alfa = learning rate
                output = self.forwardpass(X)

                E = Y - output #Raw error

                self.dZ3 = E * (output * (1 - output)) #Error multiplied by the sigmoid derivative results in the output gradient, calculates how much each output contributed to the total error
                self.dW2 = numpy.dot(self.X2.T, self.dZ3) #error propagation to the weights of the hidden layer X2 = weights of the hidden layer, we use the transposed matrix so the multiplication is possible










