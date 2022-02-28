import torch
import torch.nn as nn
import seaborn as sns
from sklearn.metrics import confusion_matrix

def Test(Net, test_set):
    j=0
    predictions = []
    labels = []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_set):
            predicted = Net(x.cuda())
            if torch.argmax(predicted).item() == y.item():
                j += 1
            predictions.append(torch.argmax(predicted).item())
            labels.append(y.item())
    cf_matrix = confusion_matrix(labels, predictions)
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
    ax.set_title('Confusion Matrix\n');
    ax.set_xlabel('Predictions')
    ax.set_ylabel('Labels');
    print("Acurária total de: " + str(100*j/len(test_set)) + "%")
    return ax