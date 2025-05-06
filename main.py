## Modelo está aprendendo XOR bem com 20000 épocas, 100 neurônios na camada oculta, 
# taxa de aprendizado de 0.001 e batch size de 3

import sys
from NNW_Structure import MLP
import numpy
args = sys.argv

def train_logic_ports(problem, n_hidden, learning_rate, epochs, batch_size):
    data = numpy.genfromtxt(f'./portas_logicas/problem{problem}.csv', delimiter=',', encoding='utf-8-sig')
    input = data[:, :2]
    desired_output = data[:, 2].reshape(-1, 1)

    mlp = MLP(n_input=2, n_hidden=n_hidden, n_output=1, learning_rate=learning_rate)
    mlp.train(input, desired_output, epochs, batch_size)

    print(f"\nResultados da porta {problem}:")
    for i in range(len(input)):
        prediction = mlp.predict(input[i].reshape(1, -1))
        print(f"Entrada: {input[i]} -> Saída esperada: {int(desired_output[i][0])} | Prevista: {int(prediction[0][0])}")


def main(args):
    if len(args) != 6:
        raise Exception("Uso: \"python main.py <problema> <épocas> <hidden layer> <learning rate> <batch size>\"")

    problem = args[1] 
    epochs = int(args[2])
    n_hidden = int(args[3])
    learning_rate = float(args[4])
    batch_size = int(args[5])
    if(problem == "AND" or problem == "OR" or problem == "XOR"):
        train_logic_ports(problem, n_hidden, learning_rate, epochs, batch_size)
    elif(problem == "CHAR"):   
        # Needs to implement
        pass
    else:
        print("Opção inválida. Digite um problema possível")

if __name__ == "__main__":
    main(args)