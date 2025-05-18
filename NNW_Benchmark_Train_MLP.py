import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from mpl_toolkits.mplot3d import Axes3D
from NNW_Structure import MLP

def load_data():
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
    return X, Y

def salvar_resultado(parametro, valor, tempo):
    path = "outputs/benchmark/benchmark_train_mlp.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "parametro": parametro,
        "valor": valor,
        "tempo": tempo
    }
    df = pd.DataFrame([data])
    if os.path.exists(path):
        df.to_csv(path, mode="a", index=False, header=False)
    else:
        df.to_csv(path, index=False)

def salvar_resultado_3d(parametro_x, parametro_y, valor_x, valor_y, tempo):
    path = "outputs/benchmark/benchmark_train_mlp_3d.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "param_x": parametro_x,
        "param_y": parametro_y,
        "valor_x": valor_x,
        "valor_y": valor_y,
        "tempo": tempo
    }
    df = pd.DataFrame([data])
    if os.path.exists(path):
        df.to_csv(path, mode="a", index=False, header=False)
    else:
        df.to_csv(path, index=False)

def testar_var_hidden(X, Y, hidden_values, epochs=30, lr=0.01, batch=32):
    print("Testando variação de neurônios na camada oculta")
    print("------------------------------------------------")
    for h in hidden_values:
        print(f"Testando {h} neurônios")
        model = MLP(n_input=120, n_hidden=h, n_output=26)
        start = time.time()
        model.train_mlp(X, Y, lr, epochs, batch)
        elapsed = time.time() - start
        salvar_resultado("neuronios", h, elapsed)

def testar_var_lr(X, Y, lr_values, epochs=30, hidden=128, batch=32):
    print("Testando variação da taxa de aprendizado")
    print("------------------------------------------------")
    for lr in lr_values:
        print(f"Testando taxa de aprendizado {lr}")
        model = MLP(n_input=120, n_hidden=hidden, n_output=26)
        start = time.time()
        model.train_mlp(X, Y, lr, epochs, batch)
        elapsed = time.time() - start
        salvar_resultado("taxa_aprendizado", lr, elapsed)

def testar_var_epochs(X, Y, epoch_values, hidden=128, lr=0.01, batch=32):
    print("Testando variação de épocas")
    print("------------------------------------------------")
    for ep in epoch_values:
        print(f"Testando {ep} épocas")
        model = MLP(n_input=120, n_hidden=hidden, n_output=26)
        start = time.time()
        model.train_mlp(X, Y, lr, ep, batch)
        elapsed = time.time() - start
        salvar_resultado("epocas", ep, elapsed)

def testar_3d_neuronios_epocas(X, Y, neuronios_list, epocas_list, lr=0.01, batch=32):
    print("Testando variação de neurônios e épocas")
    print("------------------------------------------------")
    for h in neuronios_list:
        for ep in epocas_list:
            print(f"Testando {h} neurônios e {ep} épocas")
            model = MLP(n_input=120, n_hidden=h, n_output=26)
            start = time.time()
            model.train_mlp(X, Y, lr, ep, batch)
            elapsed = time.time() - start
            salvar_resultado_3d("neuronios", "epocas", h, ep, elapsed)

def testar_3d_lr_epocas(X, Y, lr_list, epocas_list, hidden=128, batch=32):
    print("Testando variação de taxa de aprendizado e épocas")
    print("------------------------------------------------")
    for lr in lr_list:
        for ep in epocas_list:
            print(f"Testando taxa de aprendizado {lr} e {ep} épocas")
            model = MLP(n_input=120, n_hidden=hidden, n_output=26)
            start = time.time()
            model.train_mlp(X, Y, lr, ep, batch)
            elapsed = time.time() - start
            salvar_resultado_3d("taxa_aprendizado", "epocas", lr, ep, elapsed)

def gerar_graficos():
    df = pd.read_csv("outputs/benchmark/benchmark_train_mlp.csv")
    os.makedirs("outputs/benchmark", exist_ok=True)

    for parametro in df["parametro"].unique():
        sub = df[df["parametro"] == parametro]
        plt.figure(figsize=(8, 6))
        plt.plot(sub["valor"], sub["tempo"], marker='o')
        plt.xlabel(parametro.capitalize())
        plt.ylabel("Tempo de treinamento (s)")
        plt.title(f"Tempo de treinamento vs {parametro}")
        plt.grid(True)
        plt.savefig(f"outputs/benchmark/tempo_vs_{parametro}.png")
        plt.close()

def gerar_graficos_3d():
    df = pd.read_csv("outputs/benchmark/benchmark_train_mlp_3d.csv")
    for (px, py), sub in df.groupby(["param_x", "param_y"]):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_trisurf(sub["valor_x"], sub["valor_y"], sub["tempo"], cmap="viridis", edgecolor='none')
        ax.set_xlabel(px)
        ax.set_ylabel(py)
        ax.set_zlabel("Tempo de treinamento (s)")
        ax.set_title(f"Tempo vs {px} vs {py}")
        plt.savefig(f"outputs/benchmark/tempo_3d_{px}_{py}.png")
        plt.close()

if __name__ == "__main__":
    # Limpa arquivos antigos
    for path in ["outputs/benchmark/benchmark_train_mlp.csv", "outputs/benchmark/benchmark_train_mlp_3d.csv"]:
        if os.path.exists(path):
            os.remove(path)

    X, Y = load_data()

    hidden_values = [64, 128, 256, 512]
    lr_values = [0.001, 0.0001, 0.00001, 0.000001]
    epoch_values = [50, 100, 300, 500]

    # Gráficos 2D
    testar_var_hidden(X, Y, hidden_values, epochs=30, lr=0.01)
    testar_var_lr(X, Y, lr_values, epochs=30, hidden=128)
    testar_var_epochs(X, Y, epoch_values, hidden=128, lr=0.01)

    # Gráficos 3D
    testar_3d_neuronios_epocas(X, Y, hidden_values, epoch_values, lr=0.01)
    testar_3d_lr_epocas(X, Y, lr_values, epoch_values, hidden=128)

    # Geração dos gráficos
    gerar_graficos()
    gerar_graficos_3d()
