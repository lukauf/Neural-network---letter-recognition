import numpy
from NNW_Structure import MLP
from plots import create_confusion_matrix, print_error

problem = "NNW_Letters_Cross_Validation"

# Save the outputs in files
info_file = open(f"./outputs/weights/{problem}_Training_Weights.txt", "w", encoding="utf-8")

file = open(f"./outputs/predictions/{problem}.txt", "w", encoding="utf-8")

# MLP Parameters
n_input = 120
n_output = 26

n_hidden = 73
learning_rate = 0.001
epochs = 300
batch_size = 32
k_folds = 5

# Load Data
X_linhas = []
with open("./char_recognition/X.txt", "r", encoding="utf-8") as f:
    for line in f:
        values = [v.strip() for v in line.strip().split(",") if v.strip() != ""]
        if len(values) != 120:
            print("Line skipped due to wrong size:", len(values))
            continue
        X_linhas.append([float(x) for x in values])

X = numpy.array(X_linhas)

with open("./char_recognition/Y_letra.txt", "r", encoding="utf-8") as f:
    letters = [line.strip() for line in f]

indices = [ord(letter) - ord('A') for letter in letters]
Y = numpy.zeros((len(indices), 26))
for i, idx in enumerate(indices):
    Y[i, idx] = 1

shuffle_indices = numpy.random.permutation(len(X))
X = X[shuffle_indices]
Y = Y[shuffle_indices]

# K-Fold Cross-validation
num_samples = len(X)
sample_indices = numpy.arange(num_samples)
folds_size = num_samples // k_folds
scores = []

for k in range(k_folds):
    print(f"\n=== Fold {k+1}/{k_folds} ===")
    file.write(f"\n=== Fold {k+1}/{k_folds} ===\n")

    val_idxs = sample_indices[k * folds_size: (k + 1) * folds_size]
    train_idxs = numpy.setdiff1d(sample_indices, val_idxs)

    X_train_fold, Y_train_fold = X[train_idxs], Y[train_idxs]
    X_val, Y_val = X[val_idxs], Y[val_idxs]

    Nnw = MLP(n_input, n_hidden, n_output)

    # Initial Weights
    info_file.write(f"=== INITIAL WEIGHTS MEAN - FOLD {k+1} ===\n")
    info_file.write(f"W1: {numpy.mean(Nnw.W1)}\nW2: {numpy.mean(Nnw.W2)}\n\n")

    Nnw.train_mlp(X_train_fold, Y_train_fold, learning_rate, epochs, batch_size)

    # Pesos finais
    info_file.write(f"=== FINAL WEIGHTS MEAN {k+1} ===\n")
    info_file.write(f"W1: {numpy.mean(Nnw.W1)}\nW2: {numpy.mean(Nnw.W2)}\n\n")

    y_true, y_pred = [], []
    corrects = 0

    for x_sample, y_expected in zip(X_val, Y_val):
        output = Nnw.forwardpass(x_sample.reshape(1, -1))
        pred = numpy.argmax(output)
        real = numpy.argmax(y_expected)

        y_pred.append(pred)
        y_true.append(real)

        pred_letter = chr(pred + ord('A'))
        real_letter = chr(real + ord('A'))
        status = "CORRECT" if pred == real else "WRONG"
        print(f"Predicted = {pred_letter} | Expected = {real_letter} → {status}")
        file.write(f"Predicted = {pred_letter} | Expected = {real_letter} → {status}\n")

        if pred == real:
            corrects += 1

    acc = corrects / len(X_val)
    print(f"Fold {k+1} Acc: {corrects}/{len(X_val)} → {acc:.2%}")
    file.write(f"Fold {k+1} Acc: {corrects}/{len(X_val)} → {acc:.2%}\n")

    scores.append(acc)
    fold_id = f"cross_validation_folds/{problem}_fold_{k+1}"
    create_confusion_matrix(fold_id, y_true, y_pred, n_output)
    print_error(Nnw.errors_per_epoch, fold_id)

# FINAL MEAN ACCURACY AND STD ERROR
mean_acc = numpy.mean(scores)
std_error = numpy.std(scores)

print(f"\n=== Final Results ===")
print(f"Mean Accuracy: {mean_acc:.2%}")
print(f"Standard Error: {std_error:.2%}")
file.write(f"\nMean Accuracy: {mean_acc:.2%}\n")
file.write(f"Standard Error: {std_error:.2%}\n")

info_file.close()
file.close()
