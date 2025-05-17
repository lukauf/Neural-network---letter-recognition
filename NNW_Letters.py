import numpy
from NNW_Structure import MLP
from plots import create_confusion_matrix

problem = "NNW_Letters"

# Abrir arquivo para registrar as informações
info_file = open(f"./outputs/general_information/{problem}_Training_Weights.txt", "w", encoding="utf-8")

# File to store the outputs 
file = open(f"./outputs/predictions/{problem}.txt", "w", encoding="utf-8")

n_input = 120  # 120 pixels per image (input layer size)
n_hidden = 150  # number of neurons in the hidden layer (chosen empirically)
n_output = 26  # 26 possible output classes (letters A-Z)
learning_rate = 0.001
epochs = 120
batch_size = 32

# Lists to store true and predicted labels for the confusion matrix
y_true = []
y_pred = []

# Preparing the input data:
X_linhas = []

with open("./char_recognition/X.txt", "r", encoding="utf-8") as f:
    for line in f:
        values = [v.strip() for v in line.strip().split(",") if v.strip() != ""]

        if len(values) != 120:
            print("Line skipped because it has:", len(values), "values (expected: 120)")
            continue

        X_linhas.append([float(x) for x in values])

X = numpy.array(X_linhas)

# Preparing the labels
with open("./char_recognition/Y_letra.txt", "r", encoding="utf-8") as f:
    letters = [line.strip() for line in f]

# Convert letters (A-Z) to indices (0-25)
indices = [ord(letter) - ord('A') for letter in letters]
Y = numpy.zeros((len(indices), 26))

# One-hot encoding
for i, idx in enumerate(indices):
    Y[i, idx] = 1

# Split data into training and testing sets
X_train = X[:-130]  # All rows except the last 130 for training
Y_train = Y[:-130]
X_test = X[-130:]  # Last 130 rows for testing
Y_test = Y[-130:]

# Create neural network
Nnw = MLP(n_input, n_hidden, n_output)


info_file.write("=== PESOS INICIAIS ===\n")
info_file.write("W1:\n" + str(Nnw.W1) + "\n")
info_file.write("W2:\n" + str(Nnw.W2) + "\n\n")

Nnw.train_mlp(X_train, Y_train, learning_rate, epochs, batch_size)

info_file.write("=== PESOS FINAIS ===\n")
info_file.write("W1:\n" + str(Nnw.W1) + "\n")
info_file.write("W2:\n" + str(Nnw.W2) + "\n\n")
    
scores = 0

for x_sample, y_expected in zip(X_test, Y_test):
    output = Nnw.forwardpass(x_sample.reshape(1, -1))
    pred = numpy.argmax(output)
    real = numpy.argmax(y_expected)

    # Save predictions for the confusion matrix
    y_pred.append(pred)
    y_true.append(real)

    pred_letter = chr(pred + ord('A'))
    real_letter = chr(real + ord('A'))

    if pred == real:
        print(f"Predicted = {pred_letter} Expected = {real_letter} CORRECT")
        file.write(f"Predicted = {pred_letter} Expected = {real_letter} CORRECT\n")
        scores += 1
    else:
        print(f"Predicted = {pred_letter} Expected = {real_letter} WRONG")
        file.write(f"Predicted = {pred_letter} Expected = {real_letter} WRONG\n")

# Calculate accuracy for this test
accuracy = scores / len(X_test)
print(f"Accuracy: {accuracy:.2f}")
file.write(f"Accuracy: {accuracy:.2f}\n")

file.close()

# Generate and save confusion matrix + classification report
create_confusion_matrix(problem, y_true, y_pred)
