import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from include.PCA import *
from include.RandomForest import *
from include.plot_functions import *
from include.get_training_sets import *
from include.LogisticRegression import *
from include.MultiLayerPerceptron import *
from include.LinearDiscriminantAnalysis import *
from sklearn.feature_selection import chi2
pd.options.mode.chained_assignment = None

""" An analysis of the UCI Bank marketing dataset.

First features are explored and processed.

Then predictions are made with different algorithms
and oversampling methods (this is a skewed class problem)

Most of the plots are commented, just remove # to activate them."""


with open('data/bank_full.csv') as f:             # Read csv into pandas dataframe
    df = pd.read_csv( f, sep=';',header=0)

df = df.sample(frac=1).reset_index(drop=True)     # Shuffle data


del df['duration']     # Feature not available for future predictions, so it can't be used !!
del df['day']  # Not useful, possible end_of_month and beginning_of_month grouping discarded

# Class distribution - very skewed class !
#distr_plot(df['y'])

print('Feature engineering ..')
# Feature selection, engineering:

# Age clustering into Young, middle aged, and senior clients
# plot_age(df) # Plots age distribution of population, useful to decide grouping
df['Young'] = 0
df['Middle aged'] = 0
df['Senior'] = 0
df.loc[(df['age'] <= 32) & (df['age'] >= 18),'Young'] = 1
df.loc[(df['age'] <= 60) & (df['age'] >= 33),'Middle aged'] = 1
df.loc[df['age'] >=61,'Senior'] = 1
df.drop('age',axis=1 ,inplace=True)


# REMOVING BALANCE OUTLIERS - (Useful, plot boxplot before and after)
#balance_boxplot(df)
df = df[ (df['balance']-df['balance'].mean()).abs() < 15*df['balance'].std()]
#balance_boxplot(df)


# Contact medium analysis
# plot_contact(df) # Majority of contacts over cellphone
# crosstab_plot(labels,df['contact'])  # Useful feature, is one hot encoded
df['Celular'] = 0
df['Telephone'] = 0
df.loc[(df['contact'] == 'cellular') ,'Celular'] = 1
df.loc[(df['contact'] == 'telephone') ,'Telephone'] = 1
df.drop('contact',axis=1 ,inplace=True)

# Marital state
# crosstab_plot(labels,df['marital']) # to decide ordering integer encoding
df.marital.replace(('divorced','married','single') ,(-1,0,1),inplace=True)

df['Campaign freq'] = 0
df.loc[(df['campaign'] > 0) & (df['campaign'] < 5) ,'Campaign freq'] = 1
df.loc[(df['campaign'] >= 0) ,'Campaign freq'] = 2
df.drop('campaign',axis=1 ,inplace=True)

# converting previous contact into binary
df['Prev contact'] = 0
df.loc[(df['previous'] > 0) ,'Prev contact'] = 1
df.drop('previous',axis=1 ,inplace=True)

# converting previous successes into binary
df['Prev success'] = 0
df.loc[(df['poutcome'] == 'success') ,'Prev success'] = 1
df.drop('poutcome',axis=1 ,inplace=True)

# Converting prevoius days of last contact into groups of frequenct in contact
df['Frequency contact'] = 0
df.loc[(df['pdays'] > 0) & (df['pdays'] < 120) ,'Frequency contact'] = 1
df.loc[(df['pdays'] >= 120) & (df['pdays'] < 240) ,'Frequency contact'] = 1
df.loc[(df['pdays'] >= 240)  ,'Frequency contact'] = 3
df.drop('pdays',axis=1 ,inplace=True)

# dealing with balance:
df['Negative balance'] = 0
df['Pos balance'] = 0
df.loc[(df['balance'] < 0) ,'Negative balance'] = 1
df.loc[(df['balance'] > 0) & (df['balance'] < 2000) ,'Pos balance'] = 1
df.loc[(df['balance'] >= 2000) & (df['balance'] < 5000) ,'Pos balance'] = 2
df.loc[(df['balance'] >= 5000) & (df['balance'] < 10000) ,'Pos balance'] = 3
df.loc[(df['balance'] >= 10000)  ,'Pos balance'] = 4
df.drop('balance',axis=1,inplace=True)


# Integer encoding of months, the weird ordering is intentional !
# as to separate summer of winter in lower and higher values
df.month.replace(('may','jun','jul','aug','sep','oct','nov','dec','jan','feb','mar','apr')
                 ,(1,2,3,4,5,6,7,8,9,10,11,12),inplace=True)

# Education overview - Unkown and tertiary have the highest likelihood of buying the product.
# Ordering of integer encoding decided with crosstab analysis
df.education.replace(('primary','secondary','unknown','tertiary') ,(1,2,3,4),inplace=True)

# Housing and loan binary encoding
df.housing.replace(('yes','no') ,(1,0),inplace=True)
df.loan.replace(('yes','no') ,(1,0),inplace=True)

# Default overview - clients with default have less chances of buying the product.
df.default.replace(('yes','no','unknown') ,(1,0,0),inplace=True)


# Line of work analysis
#plot_job(df)
#crosstab_plot(labels, df['job'])
df.job.replace(('blue-collar','entrepreneur','housemaid','services','technician','unknown','self-employed','admin.'
                ,'management','unemployed','retired','student') ,(0,0,0,0,1,1,1,1,2,2,2,3),inplace=True)
#distr_plot(df['job']) # check resulting distribution of grouping


df_copy = df.copy()  # copies of dataframes since they are
df_copy2 = df.copy() # manipulated by oversampling methods

# Processed features description (mean, std, etc)
description = df_copy.describe()
# description.to_csv('description.csv') # write description to csv

# Processed discrete features correlation heatmap -------------------------------------------------------------------------
df_copy.y.replace(('yes','no') ,(1,0),inplace=True)
labs = df_copy['y'].values

# Dropping nominal features
df_copy.drop(['default','housing','loan','Young','Middle aged','Senior','Celular','Telephone','Prev contact'
              ,'Prev success','Negative balance'],axis=1,inplace=True)
del df_copy['y']
#plot_correlation(df_copy)  # A heatmap with spearman correlations of selected/processed discrete variables



# ----------- Component analysis, relevance of features for prediction ---------------------------
df_copy3 = df_copy2.copy()
del df_copy3['y']
df_copy3.drop(['month','education','marital','job','Campaign freq','Frequency contact','Pos balance'],axis=1,inplace=True)

# -------- perform principal component analysis and 3d plot dimesion reduced samples.

#perform_pca(df ,labs) #------ Most of the variance is explained by one feature

# chi squared test of independence
chi2v , _ = chi2(df_copy3.values , labs) # Very strong relation of class and Previous success ! Makes sense

# Plot of chi squared statistics for each variable, class pair. (Plotted on log10 scale)
# plot_chi2_test(chi2v ,df_copy3.columns.values.tolist(),log=True) # Plot results of chi2 test on logarithmic scale


# ----------------------------------------------------------------------------------------------------------------------------------

# Now data is ready to be inputed to algorithms
# The algorithms in 'algorithms' list will be trained on data
# Data will be oversampled/downsampled by methods in 'methods' list
# 'false' means neither oversampling nor downsampling

methods = [ 'false','repeat_drop','SMOTE','SMOTENN']
algorithms = ['Random Forest','Linear discriminant','Logistic Regression','MultiLayerPerceptron']

plt.show()
alg_scores = {}
print('Trying out all algorithms and sampling methods ... (may take a while, 12 min in my PC)')
v = 0
for algorithm in algorithms:
    scores = {}
    for method in methods:

        x_overs, y_overs, x_test, y_test , df_copy2 = get_sets(df_copy2 , oversampling=method, verbose=v)

        if algorithm == 'Random Forest':
            scores[method] = RandomForest(x_overs , y_overs , x_test , y_test, verbose=v)
        elif algorithm == 'Linear discriminant':
            scores[method] = LinearDiscriminant(x_overs, y_overs, x_test, y_test, verbose=v)
        elif algorithm == 'MultiLayerPerceptron':
            y_overs_2 = two_columnize(y_overs)
            y_test_2 = two_columnize(y_test)
            scores[method] = MultiLayerPerceptron(x_overs , y_overs_2 , x_test , y_test_2, verbose=v)
        elif algorithm == 'Logistic Regression':
            scores[method] = LogisticRegressor(x_overs , y_overs , x_test , y_test, verbose=v)
    alg_scores[algorithm] = scores

# Scores processing format
lists_dict = {}
plot_dict = {}
for algorithm in algorithms:
    lists_dict[algorithm] = list(sorted(alg_scores[algorithm].items()))

    plot_dict[algorithm] =  [ val[1] for val in lists_dict[algorithm]]
    plot_dict[algorithm] = np.vstack(plot_dict[algorithm])

    results_lab = [val[0] for val in lists_dict[algorithm]]


# Plotting of results (stored in 'alg_scores' dictionary)

index = np.arange(len(algorithms))*len(algorithms)
colors = ['r' ,'g' ,'b' ,'y']
aux = {}
b = -len(algorithms)*0.4
# -------------------------------------- Accuracy Plot -------------------------------------------------------------
for c ,algorithm in enumerate(algorithms):
    aux[algorithm] = plt.bar(index - b + c*0.8, plot_dict[algorithm][:,0] , label=algorithm, color=colors[c])
plt.ylabel('Accuracy')
plt.xticks(index, results_lab)
plt.legend()
plt.title('Classifier Accuracy')
plt.tight_layout()
plt.show()

# -------------------------------------- Recall Plot ---------------------------------------------------------------
for c ,algorithm in enumerate(algorithms):
    aux[algorithm] = plt.bar(index - 1.2 + c*0.8, plot_dict[algorithm][:,1] , label=algorithm, color=colors[c])
plt.ylabel('Recall')
plt.xticks(index, results_lab)
plt.legend()
plt.title('Classifier Recall')
plt.tight_layout()
plt.show()

# -------------------------------------- Precision Plot -------------------------------------------------------------
for c ,algorithm in enumerate(algorithms):
    aux[algorithm] = plt.bar(index - 1.2 + c*0.8, plot_dict[algorithm][:,2] , label=algorithm, color=colors[c])
plt.ylabel('Precision')
plt.xticks(index, results_lab)
plt.legend()
plt.title('Classifier Precision')
plt.tight_layout()
plt.show()

# -------------------------------------- Number of positive predictions Plot ------------------------------------------
for c ,algorithm in enumerate(algorithms):
    aux[algorithm] = plt.bar(index - 1.2 + c*0.8, plot_dict[algorithm][:,3] , label=algorithm, color=colors[c])
plt.ylabel('Number of positive labels')
plt.xticks(index, results_lab)
plt.legend()
plt.title('Classifier number of positive predictions')
plt.tight_layout()
plt.show()

print('Thats it, hope you like it :)')

