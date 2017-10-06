import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN

"""
Helper function to perform one of different under/over sampling
technique. Test data is separated here also.
"""



def get_sets(df , oversampling='false', verbose=1):

    df_untouched = df.copy() # preserving dataframe since we iterate over this function

    # Extracting test set
    test_df = df.tail(5000)
    test_df.y.replace(('yes','no') ,(1,0),inplace=True)
    test_pos_percent = sum(test_df['y'].values)/5000
    y_test = test_df['y'].values
    del test_df['y']
    x_test = test_df.values
    df = df.drop(df.tail(5000).index)


    df.y.replace(('yes', 'no'), (1, 0), inplace=True)  # One hot encoding of labels, stored in labels (1 COL)

    if oversampling == 'repeat_drop':
        df = df.sort_values('y')

        to_repeat = df.tail(df['y'].sum()-1)
        over_df = pd.concat([df ,to_repeat, to_repeat], axis=0)

        over_df = over_df.sample(frac=1).reset_index(drop=True)
        m1 = over_df.shape[0]

        for index, row in over_df.iterrows(): # This should be improved
            delete = np.random.choice([0,1] , p=[0.6 ,0.4])
            if row['y'] == 0 and delete == 1:
                over_df.drop(index, inplace=True)
        y_overs = over_df['y'].values

        if verbose == 1:
            print('From repeat drop: ',str(m1 - over_df.shape[0]), ' Number of samples eliminated')
            print(over_df['y'].sum() / over_df.shape[0] * 100, ' Percentage of positive labels')
            print(over_df.shape[0], ' Number of samples\n')
        del over_df['y']

        return over_df.values , y_overs, x_test , y_test ,df_untouched



    # SMOTE OVER SAMPLING
    if oversampling == 'SMOTE':
        labls = df['y'].values
        del df['y']

        X = df.values
        sm = SMOTE(random_state=42, ratio = 'minority')
        x_overs, y_overs = sm.fit_sample(X, labls)

        if verbose == 1:
            print('SMOTE oversampling results in: ',np.shape(x_overs),'Num of samples and Class ratio: ',sum(y_overs)/len(y_overs))

        return x_overs ,y_overs , x_test , y_test, df_untouched

    if oversampling == 'SMOTENN':
        labls = df['y'].values
        del df['y']

        X = df.values
        smoteen = SMOTEENN(random_state=42 )
        x_overs, y_overs = smoteen.fit_sample(X, labls)

        if verbose == 1:
            print('SMOTEEN oversampling/downsampling results in: ', np.shape(x_overs), 'Num of samples and Class ratio: ',
                  sum(y_overs) / len(y_overs))

        return x_overs ,y_overs , x_test , y_test ,df_untouched

    if oversampling == 'false':

        if verbose==1:
            print('No oversampling performed, ',df['y'].sum() / df.shape[0] * 100, ' Percentage of positive labels')

        y_overs = df['y'].values
        del df['y']

        df.drop(df.head(20000).index)

        X = df.values
        return X, y_overs, x_test, y_test ,df_untouched

    if oversampling == 'drop':
        df = df.sort_values('y')

        df.drop(df.head(30800).index,inplace=True)
        df = df.sample(frac=1).reset_index(drop=True)

        y_overs = df['y'].values
        del df['y']
        x_overs = df.values

        if verbose == 1:
            print('DROP downsampling results in: ',np.shape(x_overs),'Num of samples and Class ratio: ',sum(y_overs)/len(y_overs))

        return x_overs ,y_overs , x_test , y_test, df_untouched



