import numpy as np
from sklearn.metrics import classification_report

def meu_classification_report(y_real, y_pred):
    classes = np.unique(np.concatenate([y_real, y_pred]))

    precisoes = []
    recalls = []
    f1_scores = []
    supports = []

    for classe in classes:
        tp = np.sum((y_real == classe) & (y_pred == classe))
        fp = np.sum((y_real != classe) & (y_pred == classe))
        fn = np.sum((y_real == classe) & (y_pred != classe))

        support = tp + fn

        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        else: 0
        
        if (tp + fn) > 0:
            recall = tp / (tp + fn) 
        else: 0
        
        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall) 
        else: 0

        precisoes.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        supports.append(support)

        print(f"\nClasse {classe}:")
        print("Precision:", round(precision, 2))
        print("Recall:", round(recall, 2))
        print("F1:", round(f1, 2))
        print("Support:", support)

    accuracy = np.mean(y_real == y_pred)

    print("\nAccuracy:", round(accuracy, 2))
    print("Macro avg:", np.round([np.mean(precisoes), np.mean(recalls), np.mean(f1_scores)], 2))
    print("Weighted avg:", np.round([
        np.average(precisoes, weights=supports),
        np.average(recalls, weights=supports),
        np.average(f1_scores, weights=supports)], 2))

#teste
y_real = np.array([1, 1, 1, 1, 1, 0, 0, 0])
y_pred = np.array([1, 1, 1, 0, 0, 1, 0, 0])

meu_classification_report(y_real, y_pred)

print("\nRelatório do scikit-learn:")
print(classification_report(y_real, y_pred, zero_division=0))