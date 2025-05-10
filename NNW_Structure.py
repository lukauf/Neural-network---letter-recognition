import numpy

class MLP:
        def __init__(self, n_input, n_hidden, n_output, learning_rate): #120 input neurons, 60 (arbitrary number) neurons at hidden layer, and 26 output neurons, (26 letters/results)

                self.W1 = numpy.random.randn(n_input, n_hidden) * 0.01 #weight matrix 120x60 input layer -> hidden layer
                self.W2 = numpy.random.randn(n_hidden, n_output)  * 0.01 #weight matrix hidden layer -> output layer
                self.b1 = numpy.random.randn(1, n_hidden) #bias matrix 60x1 hidden layer bias matrix
                self.b2 = numpy.random.randn(1,n_output)
                self.learning_rate = learning_rate

        #each collumn represents the weights associated to a neuron

        #each line represents the weights which each neuron, from the previous layer, transmits to the neurons of the next layer


        @staticmethod
        def relu(z): #activation function that returns 0 if the z < 0 and returns z if z > 0, z is the total input of the neuron, sum considering the weights + the bias
                return numpy.maximum(0, z)

        @staticmethod
        def relu_derivative(z):
                return (z>0).astype(float)
        
        @staticmethod
        def sigmoid(z): #activation function that returns values between [0,1] that gets
                return 1/(1 + numpy.exp(-z))

        @staticmethod
        def sigmoid_derivative(z):
                return z * (1-z)
        
        # Hiperbolic tangent function is more recommended when we deal with logic ports with -1 and 1
        # Since sigmoid never produces values next to -1 (values in [0,1])
        @staticmethod
        def hiperbolic_tan(z):
                return (numpy.exp(z) - numpy.exp(-z))/(numpy.exp(z) + numpy.exp(-z))
        
        @staticmethod
        def hiperbolic_tan_derivative(z):
                k = MLP.hiperbolic_tan(z)
                return 1-k**2
        
        def FowardPropagation(self, X):
                # Multiply X by W1 weights and sum the bias in the hidden layer
                self.Z1 = numpy.dot(X, self.W1) + self.b1 #X(Ntx120) x W1(120x60) + b1 = Z(Ntx60)
                # Apply ReLU activation function in the hidden layer "output"
                self.A1 = self.relu(self.Z1)#X2(NtX60) the function relu is applied to every value of the array, so we get an array which every value is the return of this function that was applied on each value of the Z array
                # Multiply hidden layer "output" by W2 weights and sum the bias in the output layer
                self.Z2 = numpy.dot(self.A1, self.W2) + self.b2#X2(Ntx60) x W2(60x26) = Z2(1x26)
                # Apply sigmoid in the output
                self.A2 = self.hiperbolic_tan(self.Z2)#X3(Ntx26), X3 has the values between 0 and 1, we will later get the biggest value to see which letter is it (each slot of the array is a reference to a letter of the alfabet)
                return self.A2

        # Adjust the weights based on error between the correct ouput (Y) and the neural network calculus output (self.A2)
        def BackPropagation(self,X, Y): #first input at the network == X(Ntx120) Nt = Number of examples offered as training samples, Y(Ntx26) = Expected output, alfa = learning rate
                # Sample number (here is taking by number of "tuples")
                m = X.shape[0] 

                # Output error
                E = Y - self.A2 #Raw error
                # Output layer gradient
                dZ2 = E * self.hiperbolic_tan_derivative(self.Z2)
                # Output layer weight gradient
                dW2 = numpy.dot(self.A1.T, dZ2)
                # Output layer bias gradient 
                db2 = numpy.sum(dZ2, axis =0, keepdims=True)
                
                # Error propagated from output layer to hidden layer
                dA1 = numpy.dot(dZ2, self.W2.T)
                # Hidden layer gradient
                dZ1 = dA1 * self.relu_derivative(self.Z1)
                # Hidden layer weight gradient
                dW1 = numpy.dot(X.T, dZ1) / m
                # Hidden layer bias gradient
                db1 = numpy.sum(dZ1, axis=0, keepdims= True) / m

                #updating weights and biases
                self.W2 += self.learning_rate * dW2
                self.b2 += self.learning_rate * db2
                self.W1 += self.learning_rate * dW1
                self.b1 += self.learning_rate * db1
        
        def train(self, X, Y, epochs, batch_size):
                
                num_samples = X.shape[0]
                epoch_factor = epochs//10
                for epoch in range(epochs):
                        #array of indices that correspond to the rows of the training data
                        idxs = numpy.arange(num_samples)
                        #rearranges the elements of the indices array in a random order, so the model does not learn patterns based on the order of the data
                        numpy.random.shuffle(idxs)
                        #the rows of X are rearranged in the same random order as the indices array
                        X = X[idxs]
                        #the rows of Y are rearranged in the same random order as the indices array
                        Y= Y[idxs]

                        for start in range(0, num_samples,batch_size):
                                end = start + batch_size
                                X_batch = X[start:end]
                                Y_batch = Y[start:end]
                                
                                self.FowardPropagation(X_batch)
                                self.BackPropagation(X_batch,Y_batch)
                        
                        # For each 1000 epochs, update and print the MSE
                        if epoch %epoch_factor == 0:
                                predictions = self.predict(X)
                                loss = numpy.mean((Y - self.A2) ** 2)
                                #I'll change MSE to accuracy but I'll let it here for now
                                correct = numpy.sum(predictions == Y)
                                accuracy = correct / num_samples
                                print(f"Época {epoch} - Acc: {accuracy * 100:.2f}% - MSE: {loss:.4f}")

        def predict(self, X):
                output = self.FowardPropagation(X)
                #Values greater or equal than zeros are 1
                return numpy.where(output > 0, 1, -1)
