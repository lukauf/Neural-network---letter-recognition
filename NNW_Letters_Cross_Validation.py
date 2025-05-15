import numpy
from NNW_Structure import MLP
from confusion_matrix import create_confusion_matrix

# File to store the outputs 
file = open("./outputs/predictions/NNW_Letters_Cross_Validation_output.txt","w")

# Parâmetros da MLP
n_input = 120
n_hidden = 250
n_output = 26
learning_rate = 0.0009
epochs = 150
batch_size = 32

problem = "NNW_Letters_Cross_Validation"

# Confusion matrix
y_true = []
y_pred = []

# Folds
k_folds = 5

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

indices = [ord(letter) - ord('A') for letter in letters]  #this function uses ASCII/Unicode, so we don't need to set the index manualy

Y = numpy.zeros((len(indices), 26)) #each line of Y(expected output) will represent a one hot coded representation of a letter
for i, idx in enumerate(indices):   #fills each one hot coded line with 1 on the right spot to represent the corresponding letter
    Y[i, idx] = 1

num_samples = len(X)
sample_indices = numpy.arange(num_samples)
folds_size = num_samples // k_folds

scores = []

for k in range(k_folds):
    print(f"Fold {k+1}:")
    file.write(f"Fold {k+1}:\n")
    val_idxs = sample_indices[k * folds_size: (k + 1) * folds_size]
    train_idxs = numpy.setdiff1d(sample_indices, val_idxs)

    X_train, Y_train = X[train_idxs], Y[train_idxs]
    X_val, Y_val = X[val_idxs], Y[val_idxs]

    nnw = MLP(n_input, n_hidden, n_output)
    nnw.train_mlp(X_train, Y_train, learning_rate, epochs, batch_size)

    corrects = 0
    for x_sample, y_expected in zip(X_val, Y_val):
        output = nnw.forwardpass(x_sample.reshape(1, -1))
        pred = numpy.argmax(output) 
        real = numpy.argmax(y_expected)

        # Save the values for confusion matrix
        y_pred.append(pred)
        y_true.append(real)
        
        if pred == real:
            corrects += 1

        pred_letter = chr(pred + ord('A')) #converts the indice to letter
        real_letter = chr(real + ord('A')) #converts the indice to letter
        status = "CORRECT" if pred == real else "WRONG"
        print(f"Predicted = {pred_letter} | Expected = {real_letter} → {status}")
        file.write(f"Predicted = {pred_letter} | Expected = {real_letter} → {status}\n")
    acc = corrects / len(X_val)
    print(f"Fold {k+1} Acc: {corrects}/{len(X_val)} → {acc:.2%}")
    file.write(f"Fold {k+1} Acc: {corrects}/{len(X_val)} → {acc:.2%}\n")
    scores.append(acc)

# Final mean
print(f"\nMean accuraccy: {numpy.mean(scores):.2%}")
file.write(f"\nMean accuraccy: {numpy.mean(scores):.2%}\n")

file.close()

create_confusion_matrix(problem, y_true, y_pred)