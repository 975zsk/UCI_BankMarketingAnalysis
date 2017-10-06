from sklearn.metrics import classification_report, recall_score, precision_score, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

# Performs linear discriminant analysis on data and test it
# low variance or collinear features should be removed
def LinearDiscriminant(x_overs , y_overs , x_test , y_test , verbose=1):

    lda = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                                solver='svd', store_covariance=False, tol=0.0001)

    lda.fit(x_overs, y_overs)

    y_pred = lda.predict(x_test)

    target_names = ['No', 'Yes']
    if verbose == 1:
        print('Random Forest results:\n')
        print(classification_report(y_test, y_pred, target_names=target_names))
        print('Number of test positives is: ', sum(y_test))
        print('Number of pred positives is: ', sum(y_pred))

    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    score = [accuracy, recall, precision, sum(y_pred)]

    return np.reshape(np.array(score),[1,-1])