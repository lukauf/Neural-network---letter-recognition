import numpy
from NNW_Structure import MLP


n_input = 120 #120 pixel images
n_hidden = 60 #arbitrary
n_output = 26 #26 possible outputs to be interpreted
learning_rate = 0.01
epochs = 100
batch_size  = 32

#preparing the data:

X = numpy.loadtxt("X.txt", delimiter=",")

with open("Y_letra.txt", "r") as f:
    letters = [line.strip() for line in f]

indices = [ord(letter) - ord('A') for letter in letters] #this function uses ASCII/Unicode, so we don't need to set the index manualy

Y = numpy.zeros((len(indices), 26)) #each line of Y(expected output) will represent a one hot coded representation of a letter
for i, idx in enumerate(indices): #fills each one hot coded line with 1 on the right spot to represent the corresponding letter
    Y[i, idx] = 1

Nnw = MLP(n_input, n_hidden, n_output)

Nnw.train_mlp(X, Y, learning_rate, epochs, batch_size)

output = Nnw.forwardpass(X[0].reshape(1, -1)) #doing the test with the first letter of the matrix
predicted_index = numpy.argmax(output) #returns the index of the biggest value (1, that represents what letter is it)
predicted_letter = chr(predicted_index + ord('A')) #we get the actual letter, converting from ACII to a char

print(f"Letra prevista: {predicted_letter}")

