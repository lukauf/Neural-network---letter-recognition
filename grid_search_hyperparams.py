import numpy as np
from NNW_Structure import MLP
from sklearn.model_selection import train_test_split

# === Preparação dos dados ===
def load_data():
    print("🔄 Carregando dados...")
    X_linhas = []
    with open("./char_recognition/X.txt", "r", encoding="utf-8") as f:
        for line in f:
            values = [v.strip() for v in line.strip().split(",") if v.strip() != ""]
            if len(values) != 120:
                continue
            X_linhas.append([float(x) for x in values])
    X = np.array(X_linhas)
    print(f"✅ Dados de entrada carregados: {X.shape[0]} amostras com {X.shape[1]} atributos cada.")

    with open("./char_recognition/Y_letra.txt", "r", encoding="utf-8") as f:
        letters = [line.strip() for line in f]

    indices = [ord(letter) - ord('A') for letter in letters]
    Y = np.zeros((len(indices), 26))
    for i, idx in enumerate(indices):
        Y[i, idx] = 1

    print("✅ Labels convertidos para one-hot encoding.")
    return X, Y

# === Avaliação da acurácia ===
def evaluate(model, X_val, Y_val):
    correct = 0
    for x_sample, y_expected in zip(X_val, Y_val):
        output = model.forwardpass(x_sample.reshape(1, -1))
        pred = np.argmax(output)
        real = np.argmax(y_expected)
        if pred == real:
            correct += 1
    return correct / len(X_val)

# === Grid Search ===
def grid_search():
    try:
        X, Y = load_data()
        print("🔀 Dividindo os dados em treino e validação...")
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
        print(f"📊 Treinamento: {X_train.shape[0]} amostras | Validação: {X_val.shape[0]} amostras")

        hidden_sizes = [50, 100, 150]
        learning_rates = [0.01, 0.001]
        batch_sizes = [16, 32, 64]
        epoch_options = [50, 100, 150]

        best_accuracy = 0
        best_params = {}

        results_file = open("./outputs/grid_search_results.txt", "w", encoding="utf-8")
        results_file.write("=== GRID SEARCH - HYPERPARAMETERS ===\n\n")

        total_runs = len(hidden_sizes) * len(learning_rates) * len(batch_sizes) * len(epoch_options)
        current_run = 1

        print(f"🚀 Iniciando Grid Search: {total_runs} combinações a serem testadas...\n")

        for n_hidden in hidden_sizes:
            for lr in learning_rates:
                for bs in batch_sizes:
                    for ep in epoch_options:
                        print(f"🔧 [{current_run}/{total_runs}] Treinando com: hidden={n_hidden}, lr={lr}, batch={bs}, epochs={ep}")
                        model = MLP(n_input=120, n_hidden=n_hidden, n_output=26)
                        model.train_mlp(X_train, Y_train, lr, ep, bs)
                        acc = evaluate(model, X_val, Y_val)
                        print(f"📈 Acurácia obtida: {acc:.4f}")

                        results_file.write(f"hidden={n_hidden}, lr={lr}, batch={bs}, epochs={ep}, acc={acc:.4f}\n")

                        if acc > best_accuracy:
                            print(f"🌟 Nova melhor combinação encontrada! Acurácia: {acc:.4f}")
                            best_accuracy = acc
                            best_params = {
                                "hidden": n_hidden,
                                "learning_rate": lr,
                                "batch_size": bs,
                                "epochs": ep,
                                "accuracy": acc
                            }

                        current_run += 1
                        print("-" * 60)

        results_file.write("\n=== MELHOR COMBINAÇÃO ===\n")
        for k, v in best_params.items():
            results_file.write(f"{k}: {v}\n")
        results_file.close()

        print("\n✅ Grid Search concluído!")
        print("📁 Resultados salvos em: ./outputs/grid_search_results.txt")
        print("🏆 Melhor configuração encontrada:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    grid_search()
