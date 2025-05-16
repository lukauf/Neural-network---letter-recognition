import numpy

class MLP:
        def __init__(self, n_input, n_hidden, n_output):  # 120 input neurons, Nh (arbitrary number) neurons at hidden layer, and 26 output neurons (26 letters/results)

                self.W1 = numpy.random.randn(n_input, n_hidden) * 0.01  # weight matrix 120xNh input layer -> hidden layer
                self.W2 = numpy.random.randn(n_hidden, n_output) * 0.01  # weight matrix Nhx26 hidden layer -> output layer
                self.b1 = numpy.random.randn(1, n_hidden)  # bias matrix 1xNh hidden layer bias matrix
                self.b2 = numpy.random.randn(1, n_output)

        # each column represents the weights associated to a neuron from the next layer
        # each row represents the weights that a neuron from the current layer sends to neurons in the next layer

        @staticmethod
        def relu(z):  # activation function that returns 0 if z < 0 and returns z if z > 0, z is the total input of the neuron, sum considering the weights + the bias
                return numpy.maximum(0, z)

        @staticmethod
        def relu_derivative(z):  # returns 1 where z > 0 and 0 where z <= 0
                return (z > 0).astype(float)

        @staticmethod
        def sigmoid(z):  # activation function that returns values between [0,1]
                return 1 / (1 + numpy.exp(-z))

        @staticmethod
        def sigmoid_derivative(z):  # sigmoid derivative assumes z is already sigmoid(z)
                return z * (1 - z)

        # Hiperbolic tangent function is more recommended when we deal with logic ports with -1 and 1
        # Since sigmoid never produces values next to -1 (values in [0,1])
        @staticmethod
        def hiperbolic_tan(z):
                return (numpy.exp(z) - numpy.exp(-z)) / (numpy.exp(z) + numpy.exp(-z))

        @staticmethod
        def hiperbolic_tan_derivative(z):
                k = MLP.hiperbolic_tan(z)
                return 1 - k**2

        @staticmethod
        def swish(z):
                return z / (1 + numpy.exp(-z))  # or z * sigmoid(z)

        @staticmethod
        def swish_derivative(z):
                sigmoid_z = 1 / (1 + numpy.exp(-z))
                return sigmoid_z + z * sigmoid_z * (1 - sigmoid_z)

        def forwardpass(self, X):
                self.Z = numpy.dot(X, self.W1) + self.b1  # X(Ntx120) x W1(120xNh) + b1 = Z(NtxNh)
                self.X2 = self.swish(self.Z)  # X2(NtxNh) the function swish is applied to every value of the array, so we get an array in which every value is the result of applying swish to each value of Z
                self.Z2 = numpy.dot(self.X2, self.W2) + self.b2  # X2(NtxNh) x W2(Nhx26) + b2 = Z2(Ntx26)
                self.X3 = self.hiperbolic_tan(self.Z2)  # X3(Ntx26), X3 has the values between -1 and 1, we will later get the biggest value to see which letter it is (each slot of the array is a reference to a letter of the alphabet)
                return self.X3

        def BackPropagation(self, X, Y, alfa):  # first input at the network == X(Ntx120), Nt = Number of examples offered as training samples, Y(Ntx26) = Expected output, alfa = learning rate
                output = self.forwardpass(X)

                E = Y - output  # Raw error
                self.print_error(E)
                
                self.dZ3 = E * self.hiperbolic_tan_derivative(self.Z2)  # Error multiplied by the tanh derivative results in the output gradient, calculates how much each output activation contributed to the total error
                self.dW2 = numpy.dot(self.X2.T, self.dZ3)  # error propagation to the weights of the output layer, we use the transposed matrix so the multiplication is possible
                db2 = numpy.sum(self.dZ3, axis=0, keepdims=True)  # output bias error gradient, we sum the error of every output (each column)
                self.dA1 = numpy.dot(self.dZ3, self.W2.T)  # propagates the output error to the hidden layer
                dZ1 = self.dA1 * self.swish_derivative(self.Z)  # calculates the gradient of error (contribution of each hidden layer activation to the total error)
                self.dW1 = numpy.dot(X.T, dZ1)  # calculates the gradient of error for the weights that connect the input and hidden layer
                db1 = numpy.sum(dZ1, axis=0, keepdims=True)  # calculates the gradient of error of each bias of the hidden layer

                # Updating weights and biases using the gradients, they are updated by adding to the weights and biases the product of the learning rate and their respective gradients
                # Convergência dos pesos
                self.W1 += alfa * self.dW1
                self.b1 += alfa * db1
                self.W2 += alfa * self.dW2
                self.b2 += alfa * db2

        def train_mlp(self, X_train, Y_train, learning_rate, epochs, batch_size):  # model = MLP class instance, X_train = training data, Y_train = training labels, learning_rate = alfa(backpropagation), epochs = number of times the entire dataset will be passed through the neural network, batch_size = the number of samples to be used on the training (number of rows in X_train)
                num_samples = X_train.shape[0]
                for epoch in range(epochs):
                        indices = numpy.arange(num_samples)  # array of indices that correspond to the rows of the training data
                        numpy.random.shuffle(indices)  # rearranges the elements of the indices array in a random order, so the model does not learn patterns based on the order of the data
                        X_train = X_train[indices]  # the rows of X_train are rearranged in the same random order as the indices array
                        Y_train = Y_train[indices]  # the rows of Y_train are rearranged in the same random order as the indices array

                        for start in range(0, num_samples, batch_size):  # there will be weights changes at every iteration based on the training using the number of samples in the batch
                                end = start + batch_size  # So if the batch_size = 30, we will go from 0-30, 30-60, 60-90...
                                X_batch = X_train[start:end]
                                Y_batch = Y_train[start:end]
                                self.BackPropagation(X_batch, Y_batch, learning_rate)

        def print_error(self, E):
                error_file = open("./outputs/error.txt", "w")
                error_file.write("=== ERRO DA CAMADA DE SAÍDA ===\n")
                error_file.write(f"E: {str(E)}\n")