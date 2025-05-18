import numpy as np
import pandas as pd
from NNW_Structure import MLP
import os
import time

# === Preparação dos dados ===
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

        hidden_sizes = [73, 56, 106] #regra da média, regra do produto, regra do 2/3
        learning_rates = [0.001, 0.0001]
        batch_sizes = [16, 32]
        epoch_options = [100, 300, 500]
        patiences = [5, 10]
        k_folds_options = [3, 5]

        X_test, Y_test = X[:130], Y[:130]
        X_train_full, Y_train_full = X[130:], Y[130:]

        # === Parte 1: Treinamento padrão ===
        print("\n=== TREINAMENTO PADRÃO ===\n")
        best_params_train = {}
        best_acc_train = 0
        current_run = 1
        start_time = time.time()
        train_results = []

        total_runs = len(hidden_sizes) * len(learning_rates) * len(batch_sizes) * len(epoch_options)

        for h in hidden_sizes:
            for lr in learning_rates:
                for bs in batch_sizes:
                    for ep in epoch_options:
                        print(f"[{current_run}/{total_runs}] hidden={h}, lr={lr}, batch={bs}, epochs={ep}")
                        model = MLP(n_input=120, n_hidden=h, n_output=26)
                        model.train_mlp(X_train_full, Y_train_full, lr, ep, bs)
                        acc = evaluate(model, X_test, Y_test)

                        train_results.append({
                            "hidden": h, "lr": lr, "batch": bs, "epochs": ep, "accuracy": acc
                        })

                        if acc > best_acc_train:
                            best_acc_train = acc
                            best_params_train = {"hidden": h, "lr": lr, "batch": bs, "epochs": ep}
                        current_run += 1

        elapsed = time.time() - start_time
        minutes_train, seconds_train = divmod(elapsed, 60)

        df_train = pd.DataFrame(train_results).sort_values(by="accuracy", ascending=False)
        print("\nTabela de Resultados (Padrão):")
        print(df_train)

        print(f"\nMelhor config. (padrão): {best_params_train} com acc={best_acc_train:.4f}")
        print(f"Tempo total: {int(minutes_train)}:{int(seconds_train):02d} (min:seg)")

        # === Parte 2: Parada Antecipada ===
        print("\n=== TREINAMENTO COM PARADA ANTECIPADA ===\n")
        best_params_early = {}
        best_acc_early = 0
        current_run = 1
        start_time = time.time()
        early_results = []

        num_samples = len(X_train_full)
        train_end = int(0.9 * num_samples)
        X_train, Y_train = X_train_full[:train_end], Y_train_full[:train_end]
        X_val, Y_val = X_train_full[train_end:], Y_train_full[train_end:]

        total_runs = len(hidden_sizes) * len(learning_rates) * len(batch_sizes) * len(epoch_options) * len(patiences)

        for h in hidden_sizes:
            for lr in learning_rates:
                for bs in batch_sizes:
                    for ep in epoch_options:
                        for patience in patiences:
                            print(f"[{current_run}/{total_runs}] hidden={h}, lr={lr}, batch={bs}, max_epochs={ep}, patience={patience}")

                            model = MLP(n_input=120, n_hidden=h, n_output=26)
                            best_val_loss = float('inf')
                            patience_counter = 0
                            best_weights = None

                            for epoch in range(ep):
                                model.train_mlp(X_train, Y_train, lr, 1, bs)
                                val_outputs = model.forwardpass(X_val)
                                val_loss = np.mean((Y_val - val_outputs) ** 2)

                                if val_loss < best_val_loss:
                                    best_val_loss = val_loss
                                    patience_counter = 0
                                    best_weights = {
                                        'W1': model.W1.copy(),
                                        'b1': model.b1.copy(),
                                        'W2': model.W2.copy(),
                                        'b2': model.b2.copy()
                                    }
                                else:
                                    patience_counter += 1
                                    if patience_counter >= patience:
                                        break

                            if best_weights:
                                model.W1 = best_weights['W1']
                                model.b1 = best_weights['b1']
                                model.W2 = best_weights['W2']
                                model.b2 = best_weights['b2']

                            acc = evaluate(model, X_test, Y_test)
                            early_results.append({
                                "hidden": h, "lr": lr, "batch": bs, "max_epochs": ep, "patience": patience, "accuracy": acc
                            })

                            if acc > best_acc_early:
                                best_acc_early = acc
                                best_params_early = {
                                    "hidden": h, "lr": lr, "batch": bs, "max_epochs": ep, "patience": patience
                                }
                            current_run += 1

        elapsed = time.time() - start_time
        minutes_early, seconds_early = divmod(elapsed, 60)

        df_early = pd.DataFrame(early_results).sort_values(by="accuracy", ascending=False)
        print("\nTabela de Resultados (Early Stopping):")
        print(df_early)

        print(f"\nMelhor config. (early stopping): {best_params_early} com acc={best_acc_early:.4f}")
        print(f"Tempo total: {int(minutes_early)}:{int(seconds_early):02d} (min:seg)")

        # === Parte 3: Validação Cruzada ===
        print("\n=== VALIDAÇÃO CRUZADA ===\n")
        best_params_cv = {}
        best_acc_cv = 0
        current_run = 1
        start_time = time.time()
        cv_results = []

        total_runs = len(hidden_sizes) * len(learning_rates) * len(batch_sizes) * len(epoch_options) * len(k_folds_options)

        for h in hidden_sizes:
            for lr in learning_rates:
                for bs in batch_sizes:
                    for ep in epoch_options:
                        for k_folds in k_folds_options:
                            print(f"[{current_run}/{total_runs}] hidden={h}, lr={lr}, batch={bs}, epochs={ep}, k_folds={k_folds}")

                            scores = []
                            num_samples = len(X_train_full)
                            sample_indices = np.arange(num_samples)
                            fold_size = num_samples // k_folds

                            for k in range(k_folds):
                                val_idxs = sample_indices[k * fold_size:(k + 1) * fold_size]
                                train_idxs = np.setdiff1d(sample_indices, val_idxs)

                                X_ktrain, Y_ktrain = X_train_full[train_idxs], Y_train_full[train_idxs]

                                model = MLP(n_input=120, n_hidden=h, n_output=26)
                                model.train_mlp(X_ktrain, Y_ktrain, lr, ep, bs)

                                acc = evaluate(model, X_test, Y_test)
                                scores.append(acc)

                            mean_acc = np.mean(scores)
                            cv_results.append({
                                "hidden": h, "lr": lr, "batch": bs, "epochs": ep, "k_folds": k_folds, "mean_accuracy": mean_acc
                            })

                            if mean_acc > best_acc_cv:
                                best_acc_cv = mean_acc
                                best_params_cv = {"hidden": h, "lr": lr, "batch": bs, "epochs": ep, "k_folds": k_folds}

                            current_run += 1

        elapsed = time.time() - start_time
        minutes_cv, seconds_cv = divmod(elapsed, 60)

        df_cv = pd.DataFrame(cv_results).sort_values(by="mean_accuracy", ascending=False)
        print("\nTabela de Resultados (Cross-Validation):")
        print(df_cv)

        print(f"\nMelhor config. (cross-validation): {best_params_cv} com acc={best_acc_cv:.4f}")
        print(f"Tempo total: {int(minutes_cv)}:{int(seconds_cv):02d} (min:seg)")

        # Salvando resumo final
        results_dir = "./outputs/"
        os.makedirs(results_dir, exist_ok=True)
        results = [
            "========== RESULTADOS DO GRID SEARCH ==========",
            "",
            ">>> Melhor configuração (Treinamento Padrão):",
            f"    Parâmetros: {best_params_train}",
            f"    Acurácia:   {best_acc_train:.4f}",
            f"    Tempo:      {int(minutes_train)}:{int(seconds_train):02d} (min:seg)",
            "",
            ">>> Melhor configuração (Early Stopping):",
            f"    Parâmetros: {best_params_early}",
            f"    Acurácia:   {best_acc_early:.4f}",
            f"    Tempo:      {int(minutes_early)}:{int(seconds_early):02d} (min:seg)",
            "",
            ">>> Melhor configuração (Cross-Validation):",
            f"    Parâmetros: {best_params_cv}",
            f"    Acurácia média: {best_acc_cv:.4f}",
            f"    Tempo:      {int(minutes_cv)}:{int(seconds_cv):02d}",
            "",
            "==============================================="
        ]
        results_path = os.path.join(results_dir, "grid_search_results.txt")
        with open(results_path, "w", encoding="utf-8") as f:
            f.write("\n".join(results) + "\n")

    except Exception as e:
        print(f"Erro durante o grid search: {e}")


if __name__ == "__main__":
    grid_search()
