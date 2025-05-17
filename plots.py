from sklearn.metrics import confusion_matrix, classification_report
import seaborn
import matplotlib.pyplot

def create_confusion_matrix(problem, y_true, y_pred):
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
    print(classification_report(y_true, y_pred, target_names=[chr(i + ord('A')) for i in range(26)]))
    file.write(classification_report(y_true, y_pred, target_names=[chr(i + ord('A')) for i in range(26)]))
    file.write("\n\nPrecision: How many times the prediction was right (for each letter)\n")
    file.write("Recall: How many times the model was right when the real letter was this one (for each letter)\n")
    file.write("F1-Score: Harmonic mean between precision and recall\n")
    file.write("Support: Real samples (of each letter)\n")
    file.write("Macro Average: Metrics simple mean (all letters have the same weight)\n")
    file.write("Weighted Average: mean based on letter support\n") 
    file.close()
    # Visualization
    matplotlib.pyplot.figure(figsize=(12, 10))
    seaborn.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[chr(i + ord('A')) for i in range(26)],
                yticklabels=[chr(i + ord('A')) for i in range(26)])
    matplotlib.pyplot.xlabel('Predicted')
    matplotlib.pyplot.ylabel('Real')
    matplotlib.pyplot.title('Confusion Matrix - Letter Recognition')
    matplotlib.pyplot.savefig(f"./outputs/confusion_matrix/{problem}_confusion_matrix.png")
