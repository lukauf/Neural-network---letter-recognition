import numpy
from NNW_Structure import MLP

n_input = 120  #120 pixel images
n_hidden = 250  #arbitrary
n_output = 26  #26 possible outputs to be interpreted
learning_rate = 0.001
epochs = 120
batch_size = 32

#preparing the data:

X_linhas = []

with open("./char_recognition/X.txt", "r") as f:
    for line in f:
        # Remove spaces, separates by "," and remove empty string
        values = [v.strip() for v in line.strip().split(",") if v.strip() != ""]

        if len(values) != 120:
            print("Line skiped because it has:", len(values), "values (expected: 120)")
            continue

        X_linhas.append([float(x) for x in values])

X = numpy.array(X_linhas)

with open("./char_recognition/Y_letra.txt", "r") as f:
    letters = [line.strip() for line in f]

indices = [ord(letter) - ord('A') for letter in
           letters]  #this function uses ASCII/Unicode, so we don't need to set the index manualy

Y = numpy.zeros(
    (len(indices), 26))  #each line of Y(expected output) will represent a one hot coded representation of a letter
for i, idx in enumerate(
        indices):  #fills each one hot coded line with 1 on the right spot to represent the corresponding letter
    Y[i, idx] = 1

X_train = X[:-130]  # All lines besides the last 130
Y_train = Y[:-130]

X_test = X[-130:]  # just the last 130 lines
Y_test = Y[-130:]

Nnw = MLP(n_input, n_hidden, n_output)

Nnw.train_mlp(X_train, Y_train, learning_rate, epochs, batch_size)

scores = 0

for x_sample, y_expected in zip(X_test, Y_test):
    output = Nnw.forwardpass(x_sample.reshape(1, -1))
    pred = numpy.argmax(output)
    real = numpy.argmax(y_expected)
    pred_letter = chr(pred + ord('A'))#converts the indice to letter
    real_letter = chr(real + ord('A'))#converts the indice to letter

    if pred == real:
        print("Predicted = ", pred_letter, "Expected = ", real_letter, "CORRECT")
    else:
        print("Predicted = ", pred_letter, "Expected = ", real_letter, "WRONG")

    if pred == real:
        scores += 1

print(f"Acurácia: {scores}/{len(X_test)} → {scores / len(X_test):.2%}")