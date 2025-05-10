## Modelo está aprendendo XOR bem com 20000 épocas, 100 neurônios na camada oculta, 
# taxa de aprendizado de 0.001 e batch size de 3

import sys
from NNW_Structure import MLP
import numpy
import pandas
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

def train_char_recognition(n_hidden, learning_rate, epochs, batch_size):
    ## Sim, é o código do Lucas kkkkkkkkk
    n_input = 120
    n_output = 26

    # Preparing the data:
    with open('./caracteres/caracteres-completo/X.txt', 'r') as f:
        lines = f.readlines()

    data = []

    for line in lines:
        # Now we break the columns
        columns = [value.strip() for value in line.strip().split(',')]
    
        # Remove column 120 (X[][120])
        if len(columns) > 120:
            del columns[120]

        # Values except ' ' are now integers
        data.append([int(x) for x in columns if x != ''])

    # Converte para array numpy
    X = numpy.array(data)

    print(f"X.shape = {X.shape}")

    with open("./caracteres/caracteres-completo/Y_letra.txt", "r") as f:
        letters = [line.strip() for line in f]    
    #this function uses ASCII/Unicode, so we don't need to set the index manually
    indices = [ord(letter) - ord('A') for letter in letters]

    # Each line of Y(expected output) will represent a one hot coded representation of a letter
    
    ## Here I have a problem that I'm trying to solve
    # Y = numpy.zeros((len(indices), 26)): Array full of 0s. The training uses -1s ans 1 instead of 0s and 1s. The Acc and MSE have good values but it shows as never predicted correctly
    # Y = numpy.full((len(indices), 26), -1): Array full of -1s. The predicted values are most of the time correct but the Acc and MSE values are really strange
    Y = numpy.zeros((len(indices), 26))
    
    # Fills each one hot coded line with 1 on the right spot to represent the corresponding letter 
    for i, idx in enumerate(indices): 
        Y[i, idx] = 1 

    mlp = MLP(n_input, n_hidden, n_output, learning_rate)
    mlp.train(X, Y, epochs, batch_size)
    
    #len(indices) está errado. Mas por qual valor eu deveria trocar
    for i in range(len(indices)):
        output = mlp.FowardPropagation(X[i].reshape(1,-1))
        # Index of the biggest value
        predicted_index = numpy.argmax(output)
        # Actual letter converted from ASCII to a char
        predicted_letter = chr(predicted_index + ord('A'))
        print(f"Saída esperada: {letters[i]} - Saída prevista: {predicted_letter}")
    

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
        train_char_recognition(n_hidden, learning_rate, epochs, batch_size)
    else:
        print("Opção inválida. Digite um problema possível")

if __name__ == "__main__":
    main(args)