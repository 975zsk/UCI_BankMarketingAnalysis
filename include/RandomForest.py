from sklearn.metrics import classification_report, recall_score, precision_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Using sklearn Random forest classifier with data

def RandomForest(x_overs , y_overs , x_test , y_test, verbose=1):
    rf = RandomForestClassifier(n_estimators=20, criterion='gini', max_features=7, max_depth=5, random_state=42,
                                 verbose=0)
    rf.fit(x_overs, y_overs)

    y_pred_overs = rf.predict(x_overs)
    y_pred = rf.predict(x_test)


    target_names = ['No', 'Yes']
    if verbose == 1:
        print('Random Forest results:\n')
        print(classification_report(y_test, y_pred, target_names=target_names))
        print('Number of test positives is: ', sum(y_test))
        print('Number of pred positives is: ', sum(y_pred))

    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    score = [accuracy, recall, precision ,sum(y_pred)]

    return np.reshape(np.array(score),[1,-1])