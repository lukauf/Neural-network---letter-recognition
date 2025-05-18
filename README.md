# Neural-network---letter-recognition
Multi Layer perceptron Neural Network  trained to recognize letters 

## How to use
You can use the make {command} to run the Multilayer Perceptron Algorithm for the possible problems. <br/>
Here's the options:<br/>
<ul>
<li>make install - install the requirements</li>
<li>make letters-training - letter recognition (train only)</li>
<li>make letters - letter recognition (train + test)</li>
<li>make letters-cross-validation - letter recognition with cross validation (train + test)</li>
<li>make letters-early-stopping - letter recognition with early stopping (train + test)</li>
<li>make logic-gates AND - AND logic gate outputs</li>
<li>make logic-gates OR - OT logic gate outputs</li>
<li>make logic-gates XOR - XOR logic gate outputs</li>
<li>make letters-early-stopping-noise - letter recognition with early stopping and noise (train + test)</li>
</ul> 

## Coisa para fazer
Itens faltantes trabalho MLP:
- Grid Search: busca melhores parâmetros
- Gráficos para melhor representação - Parada antecipada | Grid Search
- Mostrar convergência de pesos (Early Stopping)
- Outro caso de validação cruzada -> considerando realizar parada antecipada
- Melhorar como mostrar erro
- Classe apontada para as entradas (no ínicio, durante e no fim do treinamento) ???

OBS:
- Removi o hiperparâmetro "número de testes"