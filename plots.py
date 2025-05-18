import numpy
from sklearn.metrics import confusion_matrix, classification_report
import seaborn
import matplotlib.pyplot as plt

def create_confusion_matrix(problem, y_true, y_pred, n_output):
    file = open(f"./outputs/classification_reports/{problem}_classification_report.txt", "w")
    
    cm = confusion_matrix(y_true, y_pred)
    
    ## Classification report:
    # Precision: How many times the prediction was right (for each letter)
    # Recall: How many times the model was right when the real letter was this one (for each letter)
    # F1-Score: Harmonic mean between precision and recall
    # Support: Real samples (of each letter)
    # Macro Average: Metrics simple mean (all letters have the same weight)
    # Weighted Average: mean based on letter support

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=[chr(i + ord('A')) for i in range(n_output)]))
    file.write(classification_report(y_true, y_pred, target_names=[chr(i + ord('A')) for i in range(n_output)]))
    file.write("\n\nPrecision: How many times the prediction was right (for each letter)\n")
    file.write("Recall: How many times the model was right when the real letter was this one (for each letter)\n")
    file.write("F1-Score: Harmonic mean between precision and recall\n")
    file.write("Support: Real samples (of each letter)\n")
    file.write("Macro Average: Metrics simple mean (all letters have the same weight)\n")
    file.write("Weighted Average: mean based on letter support\n") 
    file.close()
    # Visualization
    plt.figure(figsize=(12, 10))
    seaborn.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[chr(i + ord('A')) for i in range(n_output)],
                yticklabels=[chr(i + ord('A')) for i in range(n_output)])
    plt.xlabel('Predicted')
    plt.ylabel('Real')
    plt.title('Confusion Matrix - Letter Recognition')
    plt.savefig(f"./outputs/confusion_matrix/{problem}_confusion_matrix.png")

def print_error(errors_per_epoch, problem):
    plot_error_distribution(errors_per_epoch, problem)
    plot_loss_curve(errors_per_epoch, problem)

def plot_loss_curve(errors_mean, problem):
    plt.figure(figsize=(10, 5))
    plt.plot(errors_mean)
    plt.title(f"Mean Error per epoch - {problem}")
    plt.xlabel("Epoch")
    plt.ylabel("Mean of Errors")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./outputs/errors/loss_curve/{problem}_loss_curve.png")



def plot_error_distribution(errors_per_epoch, problem):
        flat_errors = numpy.array(errors_per_epoch).flatten()
        plt.hist(flat_errors, bins=20, edgecolor='black')
        plt.title("Mean of errors in the output Layer")
        plt.xlabel("Error")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"./outputs/errors/histogram/{problem}_error_histogram.png")
        plt.close()
