import sys
from NNW_Structure import MLP
import numpy

args = sys.argv

problem = args[1]

# File to store the outputs 
file = open(f"./outputs/NNW_Logic_Gates_{problem}_output.txt","w", encoding="utf-8")

n_hidden = 200 # AND and OR need lower values
epochs = 10000 # AND and OR need lower values
batch_size = 4
learning_rate = 0.01


data = numpy.genfromtxt(f'./logic_gates/problem{problem}.csv', delimiter=',', encoding='utf-8-sig')
input = data[:, :2]
desired_output = data[:, 2].reshape(-1, 1)
mlp = MLP(2, n_hidden,1)
mlp.train_mlp(input, desired_output, learning_rate,epochs, batch_size)
right_ones = 0
print(f"\nResultados da porta {problem}:")
for i in range(len(input)):
    # Since it's binary, check where there's 1 and -1 (based if the value in forwardpass is greater than zero)
    prediction = numpy.where(mlp.forwardpass(input[i].reshape(1, -1)) > 0, 1, -1)
    print(f"Entrada: {input[i]} -> Saída esperada: {int(desired_output[i][0])} | Prevista: {int(prediction[0][0])}")
    file.write(f"Entrada: {input[i]} -> Saída esperada: {int(desired_output[i][0])} | Prevista: {int(prediction[0][0])}\n")
    
    if int(desired_output[i][0]) == int(prediction[0][0]):
            right_ones +=1
    
print(f"Acurácia: {right_ones}/{len(input)} → {right_ones / len(input):.2%}")
file.write(f"Acurácia: {right_ones}/{len(input)} → {right_ones / len(input):.2%}\n")

file.close()