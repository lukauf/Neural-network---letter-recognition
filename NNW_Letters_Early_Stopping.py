import numpy
from NNW_Structure import MLP

n_input = 120  #120 pixel images
n_hidden = 250  #arbitrary
n_output = 26  #26 possible outputs to be interpreted
learning_rate = 0.001
epochs = 200
batch_size = 32

# patience
patience = 10
patience_counter = 0

best_val_loss = float('inf')

best_weights = None

X_linhas = []

with open("X.txt", "r") as f:
    for line in f:
        # Remove spaces, separates by "," and remove empty string
        values = [v.strip() for v in line.strip().split(",") if v.strip() != ""]

        if len(values) != 120:
            print("Line skiped because it has:", len(values), "values (expected: 120)")
            continue

        X_linhas.append([float(x) for x in values])

X = numpy.array(X_linhas)

with open("Y_letra.txt", "r") as f:
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

    val_outputs = Nnw.forwardpass(X_val)

    # MSE
    val_loss = numpy.mean((Y_val - val_outputs) ** 2)

    print(f"Epoch: {epoch+1} - Value loss: {val_loss:.6f}")

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

        if patience_counter >= patience:
            print("Early Stopping!")
            break
if best_weights:
    Nnw.W1 = best_weights['W1']
    Nnw.b1 = best_weights['b1']
    Nnw.W2 = best_weights['W2']
    Nnw.b2 = best_weights['b2']
