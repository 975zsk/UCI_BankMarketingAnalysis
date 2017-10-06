from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

""" Helper function to perform principal component analysis on data,
and plot dimension reduced data to a 3D plot.
"""

def perform_pca(df ,labels):

    dfu = df.copy()
    del dfu['y']
    X = dfu.values
    pca = PCA(n_components=3) # np.shape(X)[1]
    pca.fit(X)

    # Plot explained variance ratio of principal components
    #plt.plot(pca.explained_variance_ratio_)

    X_new = pca.transform(X)
    X_new = X_new[0:500]

    # 3D plot of dimension reduced data.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = ['r' if i == 0 else 'g' for i in labels]
    markers = ['^' if i == 0 else 'o' for i in labels]

    for x, y, z, c, m in zip(X_new[:,0], X_new[:,1] , X_new[:,2], colors, markers):
        ax.scatter(x, y, z, alpha=0.8, c=c, marker=m)

    ax.set_xlabel('Principal component 1')
    ax.set_ylabel('Principal component 2')
    ax.set_zlabel('Principal component 3')

    plt.show()

