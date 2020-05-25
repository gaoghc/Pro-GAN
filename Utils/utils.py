import tensorflow as tf


import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import normalize




def small_trick(y_test, y_pred):
    y_pred_new = np.zeros(y_pred.shape, np.bool)
    sort_index = np.flip(np.argsort(y_pred, axis=1), 1)
    for i in range(y_test.shape[0]):
        num = np.sum(y_test[i])
        for j in range(num):
            y_pred_new[i][sort_index[i][j]] = True
    return y_pred_new



def multi_label_classification(X, Y, ratio):

    X = preprocessing.normalize(X, norm='l2')

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=ratio, random_state=42)

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    #=========train=========
    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg), param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)  #
    clf.fit(X_train, y_train)
    # print('Best parameters')
    # print(clf.best_params_)

    #=========test=========
    y_pred = clf.predict_proba(X_test)
    y_pred = small_trick(y_test, y_pred)

    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")
    # print("micro_f1: %.4f" % (micro))
    # print("macro_f1: %.4f" % (macro))

    return micro, macro


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)



################################# Logging #################################
import os
import time
import sys
class Logger(object):
    def __init__(self, output_file_path):
        self.terminal = sys.stdout
        self.log = open(output_file_path, "w+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass



def logging(file_name, output_path, verbose=0):
    output_file_name = file_name.split('.')[0] + '_' + time.strftime("%d-%m-%Y_") + time.strftime("%H-%M-%S_") + "log.txt"
    output_file_path = os.path.join(output_path, output_file_name)
    sys.stdout = Logger(output_file_path)
    if verbose > 1:
        print(sys.argv)

    return output_file_path