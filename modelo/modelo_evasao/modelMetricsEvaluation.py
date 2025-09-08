import trainModule as train_mod_set
import joblib as jb
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

#cria e reporta dicionário com as métricas do modelo
def report_metrics(model_set=None, model=None, show_confusion_matrix=False):
    c_report = classification_report(y_true=model_set.y_test, y_pred=model.predict(model_set.x_test))
    
    if show_confusion_matrix:
        c_matrix = confusion_matrix(y_true=model_set.y_test, y_pred=model.predict(model_set.x_test))
        ConfusionMatrixDisplay(c_matrix).plot()
        plt.show()

    return c_report

cr = report_metrics(model_set=train_mod_set, model=jb.load(filename='LRModel.joblib'))
print(cr)


