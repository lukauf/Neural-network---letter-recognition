import numpy
from NNW_Structure import MLP

n_input = 120  # 120 pixel images
n_hidden = 200  # arbitrary
n_output = 26  # 26 possible outputs
learning_rate = 0.0009
epochs = 180
batch_size = 32
N_tests = 10  # number of tests to use for final mean calculation

# Preparing the data:
X_linhas = []

with open("./char_recognition/X.txt", "r") as f:
    for line in f:
        values = [v.strip() for v in line.strip().split(",") if v.strip() != ""]

        if len(values) != 120:
            print("Line skipped because it has:", len(values), "values (expected: 120)")
            continue

        X_linhas.append([float(x) for x in values])

X = numpy.array(X_linhas)

with open("./char_recognition/Y_letra.txt", "r") as f:
    letters = [line.strip() for line in f]

indices = [ord(letter) - ord('A') for letter in letters]
Y = numpy.zeros((len(indices), 26))

for i, idx in enumerate(indices):
    Y[i, idx] = 1

X_train = X[:-130]  # All lines besides last 130
Y_train = Y[:-130]
X_test = X[-130:]  # Just last 130 lines
Y_test = Y[-130:]

Nnw = MLP(n_input, n_hidden, n_output)

def training_and_results():
    Nnw.train_mlp(X_train, Y_train, learning_rate, epochs, batch_size)

    scores = 0

    for x_sample, y_expected in zip(X_test, Y_test):
        output = Nnw.forwardpass(x_sample.reshape(1, -1))
        pred = numpy.argmax(output)
        real = numpy.argmax(y_expected)
        pred_letter = chr(pred + ord('A'))
        real_letter = chr(real + ord('A'))

        if pred == real:
            print(f"Predicted = {pred_letter} Expected = {real_letter} CORRECT")
            scores += 1
        else:
            print(f"Predicted = {pred_letter} Expected = {real_letter} WRONG")

    accuracy = scores / len(X_test)
    print(f"Accuracy: {accuracy:.2f}")
    return accuracy

def mean_value(num_tests):
    accuracies = []  # Armazenará todas as acurácias individuais
    total_accuracy = 0.0

    for _ in range(num_tests):
        accuracy = training_and_results()
        accuracies.append(accuracy)
        total_accuracy += accuracy

    mean = total_accuracy / num_tests

    # Cálculo da variância (média dos quadrados das diferenças em relação à média)
    variance = numpy.var(accuracies, ddof=1)  # ddof=1 para variância amostral

    return mean, variance  # Retorna tanto a média quanto a variância

# Chamada atualizada:
final_mean, final_variance = mean_value(N_tests)
print(f"Final Mean: {final_mean * 100:.0f}%")
print(f"Variance: {final_variance:.4f}")