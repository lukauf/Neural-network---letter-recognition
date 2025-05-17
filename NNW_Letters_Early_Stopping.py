import numpy
from NNW_Structure import MLP
from plots import create_confusion_matrix

problem = "NNW_Letters_Early_Stopping"

# Abrir arquivo para registrar as informações
info_file = open(f"./outputs/general_information/{problem}_Training_Weights.txt", "w", encoding="utf-8")

# File to store the outputs 
file = open(f"./outputs/predictions/{problem}.txt", "w", encoding="utf-8")

n_input = 120  #120 pixel images
n_hidden = 250  #arbitrary
n_output = 26  #26 possible outputs to be interpreted
learning_rate = 0.0009
epochs = 200
batch_size = 32


# Confusion matrix
y_true = []
y_pred = []

# patience
patience = 10
patience_counter = 0

best_val_loss = float('inf')

best_weights = None

X_linhas = []

with open("./char_recognition/X.txt", "r", encoding="utf-8") as f:
    for line in f:
        # Remove spaces, separates by "," and remove empty string
        values = [v.strip() for v in line.strip().split(",") if v.strip() != ""]

        if len(values) != 120:
            print("Line skiped because it has:", len(values), "values (expected: 120)")
            continue

        X_linhas.append([float(x) for x in values])

X = numpy.array(X_linhas)

with open("./char_recognition/Y_letra.txt", "r", encoding="utf-8") as f:
    letters = [line.strip() for line in f]

indices = [ord(letter) - ord('A') for letter in
           letters]  #this function uses ASCII/Unicode, so we don't need to set the index manualy

Y = numpy.zeros(
    (len(indices), 26))  #each line of Y(expected output) will represent a one hot coded representation of a letter
for i, idx in enumerate(
        indices):  #fills each one hot coded line with 1 on the right spot to represent the corresponding letter
    Y[i, idx] = 1

num_samples = len(X)
train_end = int(0.8 * num_samples)
val_end = int(0.9 * num_samples)

# 80% for training
X_train = X[:train_end]  
Y_train = Y[:train_end]

# 10% for validation
X_val = X[train_end:val_end]
Y_val = Y[train_end:val_end]

# 10% for testing
X_test = X[val_end:]  
Y_test = Y[val_end:]

Nnw = MLP(n_input, n_hidden, n_output)

for epoch in range(epochs):
    Nnw.train_mlp(X_train, Y_train, learning_rate, 1, batch_size)

    if epoch == 0:
            info_file.write("=== PESOS INICIAIS ===\n")
            info_file.write("W1:\n" + str(Nnw.W1) + "\n")
            info_file.write("W2:\n" + str(Nnw.W2) + "\n\n")

    val_outputs = Nnw.forwardpass(X_val)

    # MSE
    val_loss = numpy.mean((Y_val - val_outputs) ** 2)

    print(f"Epoch: {epoch+1} - Value loss: {val_loss:.6f}")
    file.write(f"Epoch: {epoch+1} - Value loss: {val_loss:.6f}\n")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0

        best_weights = {
            'W1': Nnw.W1.copy(),
            'b1': Nnw.b1.copy(),
            'W2': Nnw.W2.copy(),
            'b2': Nnw.b2.copy()
        }
    else:
        patience_counter += 1
        print(f"No best results. Patience: {patience_counter}/{patience}")
        file.write(f"No best results. Patience: {patience_counter}/{patience}\n")

        if patience_counter >= patience:
            info_file.write("=== PESOS FINAIS ===\n")
            info_file.write("W1:\n" + str(Nnw.W1) + "\n")
            info_file.write("W2:\n" + str(Nnw.W2) + "\n\n")
  
            print("==============================================================")
            print(f"\tEarly Stopping! - Best value loss: {best_val_loss:6f}")
            print("==============================================================")
            file.write("==============================================================\n")
            file.write(f"\tEarly Stopping! - Best value loss: {best_val_loss:6f}\n")
            file.write("==============================================================\n")
            break
if best_weights:
    Nnw.W1 = best_weights['W1']
    Nnw.b1 = best_weights['b1']
    Nnw.W2 = best_weights['W2']
    Nnw.b2 = best_weights['b2']

for x_sample, y_expected in zip(X_test, Y_test):
    output = Nnw.forwardpass(x_sample.reshape(1, -1))
    pred = numpy.argmax(output)
    real = numpy.argmax(y_expected)

    # Save the values for confusion matrix
    y_pred.append(pred)
    y_true.append(real)

    pred_letter = chr(pred + ord('A'))#converts the indice to letter
    real_letter = chr(real + ord('A'))#converts the indice to letter

file.close()

create_confusion_matrix(problem, y_true, y_pred)