
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


def metrics(model,dataset,y_labels, file_name,results_path):

    Y_predicted = model.predict(dataset)
    Y_predicted = np.argmax(Y_predicted,axis = 1)

    cm = confusion_matrix(y_labels, Y_predicted)

    print(cm)
    print("***********Classification Report  set")

    print(classification_report(y_labels, Y_predicted))

    with open(results_path + file_name + '.txt', "w") as f:
        print(cm, file=f)
        print(classification_report(y_labels, Y_predicted), file=f)

