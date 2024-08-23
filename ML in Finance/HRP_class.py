import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import numpy as np
import pandas as pd

class HierarchicalRiskParity:
    def __init__(self, cov):
        """
        Initialize with a covariance matrix.
        """
        self.cov = cov

    def getIVP(self, cov=None):
        """
        Compute the inverse-variance portfolio.
        If no covariance matrix is provided, use the class covariance matrix.
        """
        if cov is None:
            cov = self.cov
        ivp = 1. / np.diag(cov)
        ivp /= ivp.sum()
        return ivp

    def getClusterVar(self, cItems):
        """
        Compute variance per cluster.
        """
        cov_ = self.cov.loc[cItems, cItems]  # matrix slice
        w_ = self.getIVP(cov_).reshape(-1, 1)
        cVar = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
        return cVar

    def getQuasiDiag(self, link):
        """
        Sort clustered items by distance.
        """
        link = link.astype(int)
        sortIx = pd.Series([link[-1, 0], link[-1, 1]])
        numItems = link[-1, 3]  # number of original items
        while sortIx.max() >= numItems:
            sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # make space
            df0 = sortIx[sortIx >= numItems]  # find clusters

            i = df0.index
            j = df0.values - numItems
            sortIx[i] = link[j, 0]  # item 1
            df0 = pd.Series(link[j, 1], index=i + 1)
            sortIx = pd.concat([sortIx, df0]).sort_index()  # item 2
            sortIx.index = range(sortIx.shape[0])  # re-index
        return sortIx.tolist()

    def getRecBipart(self, link):
        """
        Compute HRP allocation using linkage matrix.Here we use 'single method', ultilizing the minimum
        distance between cluster, it helps to identify the most tightly knit groups of assets, 
        which can then be diversified against other groups.
        """
        sortIx = self.getQuasiDiag(link)
        w = pd.Series(1.0, index=sortIx)  # Ensure the Series is float type
        cItems = [sortIx]  # initialize all items in one cluster
        while len(cItems) > 0:
            cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]  # bi-section
            for i in range(0, len(cItems), 2):  # parse in pairs
                cItems0 = cItems[i]  # cluster 1
                cItems1 = cItems[i + 1]  # cluster 2
                cVar0 = self.getClusterVar(cItems0)
                cVar1 = self.getClusterVar(cItems1)
                alpha = 1 - cVar0 / (cVar0 + cVar1)
                w[cItems0] *= alpha  # weight 1
                w[cItems1] *= 1 - alpha  # weight 2
        return w

    def correlDist(self, corr):
        """
        A distance matrix based on correlation, where 0<=d[i,j]<=1.
        This is a proper distance metric.
        """
        dist = ((1 - corr) / 2.) ** 0.5  # distance matrix
        return dist

    def plotCorrMatrix(self, corr, labels=None):
        """
        Heatmap of the correlation matrix.
        """
        if labels is None:
            labels = []
        plt.pcolor(corr)
        plt.colorbar()
        plt.yticks(np.arange(.5, corr.shape[0] + .5), labels)
        plt.xticks(np.arange(.5, corr.shape[0] + .5), labels)
        plt.show()
        