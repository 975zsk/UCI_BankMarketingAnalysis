import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Auxiliary function not plot
def two_columnize(y):

    y_2 = np.zeros([np.shape(y)[0] , 2])
    for i in range(0,np.shape(y)[0]):
        if y[i] == 1:
            y_2[i ,1] += 1
        else:
            y_2[i, 0] += 1
    return y_2

"""
All plots called in RunAnalysis script except for classifier performance
ones are defined here.
"""

def distr_plot(col):
    col.value_counts().plot(kind='bar')
    plt.title('Distribution of ' + col.name)
    plt.xlabel('Succesfull')
    plt.ylabel('Counts')
    plt.xticks(rotation=25)
    plt.show()

def plot_age(df):
    df['age'].plot(kind='hist',bins=40)
    plt.title('Age distribution of clients')
    plt.xlabel('Age')
    plt.show()

def balance_boxplot(df):
    df.boxplot(column='balance')
    plt.title('Balance Boxplot')
    plt.ylabel('Money ($)')
    plt.show()

def plot_contact(df):
    df['contact'].value_counts().plot(kind='bar')
    plt.title('contact type distribution ')
    plt.xlabel('contact category')
    plt.ylabel('number of samples')
    plt.show()

def plot_job(df):
    df['education'].value_counts().plot(kind='bar')
    plt.title('job distribution of clients')
    plt.xlabel('Job')
    plt.show()


def crosstab_plot(labels ,col):
    cross = pd.crosstab(labels, col).apply(lambda r: r / r.sum(), axis=0)
    toplot = cross.drop(['no'])
    toplot = ((toplot.transpose()).sort_values(['yes'], ascending=[1])).transpose()

    my_colors = []
    for column in toplot:
        if toplot[column][0] < 0.09:
            my_colors.append('#FF0000')
        elif toplot[column][0] < 0.117:
            my_colors.append('#F7FE2E')
        elif toplot[column][0] < 0.14:
            my_colors.append('#F7FE2E')
        else:
            my_colors.append('#00FF00')

    toplot.plot(kind='bar', color=my_colors)
    plt.title('Percentage of positive lables in ' + col.name)
    plt.xlabel('Categorical feature')
    plt.ylabel('Percentage of yes labels')
    plt.show()



def plot_correlation(d):
    corr = d.corr(method='spearman')

    sns.set(style="white")

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    thismanager = plt.get_current_fig_manager()
    thismanager.window.setGeometry(50,100,640, 1545)
    plt.xticks(rotation=15)
    plt.title('Correlation Heatmap of non binary variables')
    plt.show()

def plot_chi2_test(chi2v ,ticks,log=False):
    if log:
        chi2v = np.log10(chi2v)

    plt.bar(range(0,len(chi2v)),chi2v , tick_label=ticks)
    plt.xticks(rotation=35)
    plt.title('Chi squared independence test')
    plt.ylabel('Log10 of (χ²) for each variable-class')
    thismanager = plt.get_current_fig_manager()
    thismanager.window.setGeometry(50, 100, 640, 1545)
    plt.show()