import numpy as np
from NNW_Structure import MLP
from plots import create_confusion_matrix, print_error

# Configurações
problem = "NNW_Letters_CV_Early_Stopping"
k_folds = 5
n_input = 120
n_hidden = 250
n_output = 26
learning_rate = 0.0009
epochs = 600
batch_size = 32
patience = 10

# Abrir arquivos de saída
info_file = open(f"./outputs/weights/{problem}_Training_Weights.txt", "w", encoding="utf-8")
file = open(f"./outputs/predictions/{problem}.txt", "w", encoding="utf-8")

# Carregamento dos dados
X_linhas = []
with open("./char_recognition/X.txt", "r", encoding="utf-8") as f:
    for line in f:
        values = [v.strip() for v in line.strip().split(",") if v.strip() != ""]
        if len(values) != 120:
            continue
        X_linhas.append([float(x) for x in values])
X = np.array(X_linhas)

with open("./char_recognition/Y_letra.txt", "r", encoding="utf-8") as f:
    letters = [line.strip() for line in f]
indices = [ord(letter) - ord('A') for letter in letters]
Y = np.zeros((len(indices), 26))
for i, idx in enumerate(indices):
    Y[i, idx] = 1

# Separação treino/teste
X_train = X[:-130]
Y_train = Y[:-130]
X_test = X[-130:]
Y_test = Y[-130:]

# Validação cruzada
num_samples = len(X_train)
sample_indices = np.arange(num_samples)
fold_size = num_samples // k_folds
scores = []

for k in range(k_folds):
    print(f"\n=== Fold {k+1}/{k_folds} ===")
    file.write(f"\n=== Fold {k+1}/{k_folds} ===\n")

    val_idxs = sample_indices[k * fold_size: (k + 1) * fold_size]
    train_idxs = np.setdiff1d(sample_indices, val_idxs)

    X_train_fold, Y_train_fold = X_train[train_idxs], Y_train[train_idxs]
    X_val, Y_val = X_train[val_idxs], Y_train[val_idxs]

    # Inicializa MLP
    mlp = MLP(n_input, n_hidden, n_output)

    print(f"=== INITIAL WEIGHTS MEAN - FOLD {k+1}===")
    print(f"W1:\n {str(np.mean(mlp.W1))}")
    print(f"W2:\n {str(np.mean(mlp.W2))}")
    info_file.write(f"=== INITIAL WEIGHTS MEAN - FOLD {k+1}===\n")
    info_file.write(f"W1:\n {str(np.mean(mlp.W1))}\n")
    info_file.write(f"W2:\n {str(np.mean(mlp.W2))}\n")

    best_val_loss = float('inf')
    patience_counter = 0
    best_weights = None

    for epoch in range(epochs):
        mlp.train_mlp(X_train_fold, Y_train_fold, learning_rate, 1, batch_size)

        val_outputs = mlp.forwardpass(X_val)
        
        # Errors
        val_errors = np.abs(Y_val - val_outputs)
        val_loss = np.mean((Y_val - val_outputs) ** 2)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = {
                'W1': mlp.W1.copy(),
                'b1': mlp.b1.copy(),
                'W2': mlp.W2.copy(),
                'b2': mlp.b2.copy()
            }
            if epoch == epochs:
                print(f"No Early Stopping - Best MSE {best_val_loss:.6f}")
                file.write(f"No Early Stopping - Best MSE {best_val_loss:.6f}\n")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} - Best MSE: {best_val_loss:.6f}")
                file.write(f"Early stopping at epoch {epoch+1} - Best MSE: {best_val_loss:.6f}\n")
                break
    
    print(f"=== FINAL WEIGHTS MEAN - FOLD {k+1} ===")
    print("W1:\n" + str(np.mean(mlp.W1)) + "")
    print("W2:\n" + str(np.mean(mlp.W2)) + "")
    info_file.write(f"=== FINAL WEIGHTS MEAN - FOLD {k+1} ===\n")
    info_file.write("W1:\n" + str(np.mean(mlp.W1)) + "\n")
    info_file.write("W2:\n" + str(np.mean(mlp.W2)) + "\n\n")

    if best_weights:
        mlp.W1 = best_weights['W1']
        mlp.b1 = best_weights['b1']
        mlp.W2 = best_weights['W2']
        mlp.b2 = best_weights['b2']

    # Avaliação no fold atual
    corrects = 0
    y_true, y_pred = [], []
    for x_sample, y_expected in zip(X_val, Y_val):
        output = mlp.forwardpass(x_sample.reshape(1, -1))
        pred = np.argmax(output)
        real = np.argmax(y_expected)
        y_pred.append(pred)
        y_true.append(real)
        if pred == real:
            corrects += 1

    acc = corrects / len(X_val)
    print(f"Fold {k+1} Acc: {acc:.2%}")
    file.write(f"Fold {k+1} Acc: {acc:.2%}\n")
    scores.append(acc)

    problem_fold = f"cross_validation_folds/{problem}_training_fold_{k+1}"
    create_confusion_matrix(problem_fold, y_true, y_pred, n_output)
    print_error(mlp.errors_per_epoch,problem_fold)

# Avaliação Final no Teste
print("\n=== Final Evaluation on Unseen Test Data ===")
file.write("\n=== Final Evaluation on Unseen Test Data ===\n")
final_model = MLP(n_input, n_hidden, n_output)

# Treinar no conjunto completo de treino com early stopping
best_val_loss = float('inf')
patience_counter = 0
best_weights = None

train_end = int(0.9 * len(X_train))
X_train_final = X_train[:train_end]
Y_train_final = Y_train[:train_end]
X_val_final = X_train[train_end:]
Y_val_final = Y_train[train_end:]

for epoch in range(epochs):
    final_model.train_mlp(X_train_final, Y_train_final, learning_rate, 1, batch_size)
    val_outputs = final_model.forwardpass(X_val_final)
    val_loss = np.mean((Y_val_final - val_outputs) ** 2)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_weights = {
            'W1': final_model.W1.copy(),
            'b1': final_model.b1.copy(),
            'W2': final_model.W2.copy(),
            'b2': final_model.b2.copy()
        }
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Final Early stopping at epoch {epoch+1}")
            break

if best_weights:
    final_model.W1 = best_weights['W1']
    final_model.b1 = best_weights['b1']
    final_model.W2 = best_weights['W2']
    final_model.b2 = best_weights['b2']

# Avaliação final
correct_test = 0
y_true_test, y_pred_test = [], []
for x_sample, y_expected in zip(X_test, Y_test):
    output = final_model.forwardpass(x_sample.reshape(1, -1))
    pred = np.argmax(output)
    real = np.argmax(y_expected)
    y_pred_test.append(pred)
    y_true_test.append(real)
    if pred == real:
        correct_test += 1

test_acc = correct_test / len(X_test)
print(f"Test Accuracy: {test_acc:.2%}")
file.write(f"Test Accuracy: {test_acc:.2%}\n")
create_confusion_matrix("NNW_Letters_CV_Early_Stopping", y_true_test, y_pred_test, n_output)

# Estatísticas finais
print(f"\nMean Accuracy: {np.mean(scores):.2%}")
print(f"Standard Error: {np.std(scores):.2%}")
file.write(f"\nMean Accuracy: {np.mean(scores):.2%}\n")
file.write(f"Standard Error: {np.std(scores):.2%}\n")

file.close()
info_file.close()
