from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
import numpy as np
from sklearn.metrics import classification_report, recall_score, precision_score, accuracy_score
import os

# Performs logistic regression on data and tests it

def LogisticRegressor(x_overs , y_overs , x_test , y_test, verbose=1):

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    model = Sequential()
    model.add(Dense(1, input_dim=x_overs.shape[1]))
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit(x_overs, y_overs, epochs=10,verbose=0)

    y_pred = np.round(model.predict(x_test))

    target_names = ['No', 'Yes']

    if verbose == 1:
        print('Logistic Regression results:\n')
        print(classification_report(y_test, y_pred, target_names=target_names))
        print('Number of test positives is: ', sum(y_test))
        print('Number of pred positives is: ', sum(y_pred))

    y_pred = np.squeeze(y_pred)

    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test , y_pred)
    accuracy = accuracy_score(y_test , y_pred)
    score = [accuracy ,recall ,precision ,sum(y_pred)]

    return np.reshape(np.array(score),[1,-1])