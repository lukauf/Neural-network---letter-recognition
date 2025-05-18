import sys
import numpy
from NNW_Structure import MLP
from plots import create_confusion_matrix, print_error, plot_loss_curve

difficulty = sys.argv[1]
problem = "NNW_Letters_Early_Stopping"

if difficulty == "noise" or difficulty == "merge-classes":
    problem = f"{problem}_{difficulty}"

# Abrir arquivo para registrar as informações
info_file = open(f"./outputs/weights/{problem}_Training_Weights.txt", "w", encoding="utf-8")

# File to store the outputs 
file = open(f"./outputs/predictions/{problem}.txt", "w", encoding="utf-8")

n_input = 120  #120 pixel images
n_hidden = 250  #arbitrary
n_output = 26  #26 possible outputs to be interpreted
learning_rate = 0.0009
epochs = 600
batch_size = 32

# add difficulties on the test


best_values = []

val_mean_errors = []
test_mean_errors = []

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

if difficulty == "noise":
    # Adiciona ruído gaussiano com média 0 e desvio padrão 0.3
    noise = numpy.random.normal(loc=0.0, scale=0.3, size=X.shape)
    X += noise

    # Garante que os valores continuem próximos de -1 a 1 (com leve tolerância)
    X = numpy.clip(X, -1.5, 1.5)

with open("./char_recognition/Y_letra.txt", "r", encoding="utf-8") as f:
    letters = [line.strip() for line in f]

if difficulty == "merge-classes":
    # Novo mapeamento de letras com agrupamentos
    classe_map = {}
    nova_classe = 0

    agrupamentos = {
        "DO": ["D", "O"],
        "IJT": ["I", "J", "T"]
    }

    # Atribuir classes agrupadas
    for grupo, letras_grupo in agrupamentos.items():
        for letra in letras_grupo:
            classe_map[letra] = nova_classe
        nova_classe += 1

    # Atribuir classes restantes
    for letra in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if letra not in classe_map:
            classe_map[letra] = nova_classe
            nova_classe += 1

    # Atualiza a saída Y com base no novo mapeamento
    n_output = len(set(classe_map.values()))  # agora é 24
    Y = numpy.zeros((len(letters), n_output))

    for i, letra in enumerate(letters):
        idx = classe_map[letra]
        Y[i, idx] = 1
else:
    indices = [ord(letter) - ord('A') for letter in letters]  #this function uses ASCII/Unicode, so we don't need to set the index manualy

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
            info_file.write("=== MÉDIA DOS PESOS INICIAIS ===\n")
            info_file.write("W1:\n" + str(numpy.mean(Nnw.W1)) + "\n")
            info_file.write("W2:\n" + str(numpy.mean(Nnw.W2)) + "\n\n")
            print("=== MÉDIA DOS PESOS INICIAIS ===\n")
            print("W1:\n" + str(numpy.mean(Nnw.W1)) + "\n")
            print("W2:\n" + str(numpy.mean(Nnw.W2)) + "\n\n")


    val_outputs = Nnw.forwardpass(X_val)

    # Cálculo do erro médio absoluto por época
    
    # Erros
    val_errors = numpy.abs(Y_val - val_outputs)

    # Cálculo do erro médio absoluto por época
    val_abs_error_mean = numpy.mean(val_errors)
    
    val_mean_errors.append(val_abs_error_mean)

    # MSE
    val_loss = numpy.mean((Y_val - val_outputs) ** 2)

    print(f"Epoch: {epoch+1} - MSE: {val_loss:.6f}")
    file.write(f"Epoch: {epoch+1} - MSE: {val_loss:.6f}\n")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0

        best_values.append(best_val_loss)


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
            info_file.write("=== MÉDIA DOS PESOS FINAIS ===\n")
            info_file.write("W1:\n" + str(numpy.mean(Nnw.W1)) + "\n")
            info_file.write("W2:\n" + str(numpy.mean(Nnw.W2)) + "\n\n")
            print("=== MÉDIA DOS PESOS FINAIS ===\n")
            print("W1:\n" + str(numpy.mean(Nnw.W1)) + "\n")
            print("W2:\n" + str(numpy.mean(Nnw.W2)) + "\n\n")
            
            print("==============================================================")
            print(f"\tEarly Stopping! - Best MSE: {best_val_loss:6f}")
            print("==============================================================")
            file.write("==============================================================\n")
            file.write(f"\tEarly Stopping! - Best MSE: {best_val_loss:6f}\n")
            file.write("==============================================================\n")
            break
if best_weights:
    Nnw.W1 = best_weights['W1']
    Nnw.b1 = best_weights['b1']
    Nnw.W2 = best_weights['W2']
    Nnw.b2 = best_weights['b2']

problem_name = f"{problem}_training"
print_error(Nnw.errors_per_epoch, problem_name)

for x_sample, y_expected in zip(X_test, Y_test):
    output = Nnw.forwardpass(x_sample.reshape(1, -1))
    pred = numpy.argmax(output)
    real = numpy.argmax(y_expected)

    # Save the values for confusion matrix
    y_pred.append(pred)
    y_true.append(real)

    sample_error = numpy.abs(y_expected - output) 
    mean_sample_error = numpy.mean(sample_error)
    test_mean_errors.append(mean_sample_error)

    pred_letter = chr(pred + ord('A'))#converts the indice to letter
    real_letter = chr(real + ord('A'))#converts the indice to letter

file.close()

create_confusion_matrix(problem, y_true, y_pred, n_output)

val_problem = f"{problem}_validation"
plot_loss_curve(val_mean_errors, val_problem)

test_problem = f"{problem}_test"
plot_loss_curve(test_mean_errors, test_problem)
