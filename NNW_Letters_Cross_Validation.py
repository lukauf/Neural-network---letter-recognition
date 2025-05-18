import numpy
from NNW_Structure import MLP
from plots import create_confusion_matrix, print_error

problem = "NNW_Letters_Cross_Validation"

# Abrir arquivo para registrar as informações
info_file = open(f"./outputs/weights/{problem}_Training_Weights.txt", "w", encoding="utf-8")

# File to store the outputs 
file = open(f"./outputs/predictions/{problem}.txt", "w", encoding="utf-8")

# Parâmetros da MLP
n_input = 120
n_hidden = 250
n_output = 26
learning_rate = 0.0009
epochs = 150
batch_size = 32

problem = "NNW_Letters_Cross_Validation"

# Folds
k_folds = 5

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

with open("./char_recognition/Y_letra.txt", "r",encoding="utf-8") as f:
    letters = [line.strip() for line in f]

indices = [ord(letter) - ord('A') for letter in letters]  #this function uses ASCII/Unicode, so we don't need to set the index manualy

Y = numpy.zeros((len(indices), 26)) #each line of Y(expected output) will represent a one hot coded representation of a letter
for i, idx in enumerate(indices):   #fills each one hot coded line with 1 on the right spot to represent the corresponding letter
    Y[i, idx] = 1

scores = []

# Split data into training and testing sets
X_train = X[:-130]  # All rows except the last 130 for training
Y_train = Y[:-130]
X_test = X[-130:]  # Last 130 rows for testing
Y_test = Y[-130:]

num_samples = len(X_train)
sample_indices = numpy.arange(num_samples)
folds_size = num_samples // k_folds

for k in range(k_folds):
   
    # Confusion matrix
    y_true = []
    y_pred = []

    print(f"Fold {k+1}:")
    file.write(f"Fold {k+1}:\n")
    val_idxs = sample_indices[k * folds_size: (k + 1) * folds_size]
    train_idxs = numpy.setdiff1d(sample_indices, val_idxs)

    X_train_fold, Y_train_fold = X_train[train_idxs], Y_train[train_idxs]
    X_val, Y_val = X_train[val_idxs], Y_train[val_idxs]

    Nnw = MLP(n_input, n_hidden, n_output)
    
    info_file.write(f"=== MÉDIA DOS PESOS INICIAIS - FOLD {k+1}===\n")
    info_file.write("W1:\n" + str(numpy.mean(Nnw.W1)) + "\n")
    info_file.write("W2:\n" + str(numpy.mean(Nnw.W2)) + "\n\n")
    print(f"=== MÉDIA DOS PESOS INICIAIS - FOLD {k+1}===\n")
    print("W1:\n" + str(numpy.mean(Nnw.W1)) + "\n")
    print("W2:\n" + str(numpy.mean(Nnw.W2)) + "\n\n")

    Nnw.train_mlp(X_train_fold, Y_train_fold, learning_rate, epochs, batch_size)
    
    info_file.write(f"=== MÉDIA DOS PESOS FINAIS - FOLD {k+1}===\n")
    info_file.write("W1:\n" + str(numpy.mean(Nnw.W1)) + "\n")
    info_file.write("W2:\n" + str(numpy.mean(Nnw.W2)) + "\n\n")
    print(f"=== MÉDIA DOS PESOS FINAIS - FOLD {k+1}===\n")
    print("W1:\n" + str(numpy.mean(Nnw.W1)) + "\n")
    print("W2:\n" + str(numpy.mean(Nnw.W2)) + "\n\n")

    corrects = 0
    for x_sample, y_expected in zip(X_val, Y_val):
        output = Nnw.forwardpass(x_sample.reshape(1, -1))
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
    problem_fold = f"cross_validation_folds/{problem}_fold_{k+1}"
    create_confusion_matrix(problem_fold, y_true, y_pred, n_output)
    print_error(Nnw.errors_per_epoch, problem_fold)

# Avaliação final com dados de teste
print("\n=== Teste final com dados não vistos ===")
file.write("\n=== Teste final com dados não vistos ===\n")

# Treina novamente com todo o conjunto de treino
final_model = MLP(n_input, n_hidden, n_output)
final_model.train_mlp(X_train, Y_train, learning_rate, epochs, batch_size)

y_true_test = []
y_pred_test = []
correct_test = 0

for x_sample, y_expected in zip(X_test, Y_test):
    output = final_model.forwardpass(x_sample.reshape(1, -1))
    pred = numpy.argmax(output)
    real = numpy.argmax(y_expected)

    y_pred_test.append(pred)
    y_true_test.append(real)

    pred_letter = chr(pred + ord('A'))
    real_letter = chr(real + ord('A'))
    status = "CORRECT" if pred == real else "WRONG"
    print(f"Predicted = {pred_letter} | Expected = {real_letter} → {status}")
    file.write(f"Predicted = {pred_letter} | Expected = {real_letter} → {status}\n")
    
    if pred == real:
        correct_test += 1

test_acc = correct_test / len(X_test)
print(f"Test Accuracy: {correct_test}/{len(X_test)} → {test_acc:.2%}")
file.write(f"Test Accuracy: {correct_test}/{len(X_test)} → {test_acc:.2%}\n")

create_confusion_matrix(problem, y_true_test, y_pred_test, n_output)


# Final mean
print(f"\nMean accuraccy: {numpy.mean(scores):.2%}")
file.write(f"\nMean accuraccy: {numpy.mean(scores):.2%}\n")
print(f"\nStandard Error: {numpy.std(scores):.2%}")
file.write(f"\nStandard Error: {numpy.std(scores):.2%}\n")
file.close()