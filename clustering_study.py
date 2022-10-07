import numpy as np
import skfuzzy as fuzz
import pandas as pd
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from src.plotCorrelationMatrix import plotCorrelationMatrix
from algorithms_analisys import NestedKFoldAlgoTester
data = pd.read_csv('smart_grid_stability_augmented.csv')
data['stabf'] = data['stabf'].apply(lambda x: x == 'stable')

X = data.iloc[:, :12]
y = data.stabf

tau = X.iloc[:, :4]
p = X.iloc[:, 4:8]
g = X.iloc[:, 8:]

node1 = X.iloc[:, [0, 4, 8]]
node2 = X.iloc[:, [1, 5, 9]]
node3 = X.iloc[:, [2, 6, 10]]
node4 = X.iloc[:, [3, 7, 11]]


if __name__ == "__main__":
    # fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))
    fpcs1 = []
    fpcs2 = []
    fpcs3 = []
    fpcs4 = []
    x = []
    fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))

    for ncenters, ax in enumerate(axes1.reshape(-1), 2):
        x.append(ncenters)

        cntr, u, u0, d, jm, p, fpc1 = fuzz.cluster.cmeans(
            node1, ncenters, 2, error=0.005, maxiter=1000, init=None)

        # Store fpc values for later
        fpcs1.append(fpc1)

        cntr, u, u0, d, jm, p, fpc2 = fuzz.cluster.cmeans(
            node2, ncenters, 2, error=0.005, maxiter=1000, init=None)

        # Store fpc values for later
        fpcs2.append(fpc2)

        cntr, u, u0, d, jm, p, fpc3 = fuzz.cluster.cmeans(
            node3, ncenters, 2, error=0.005, maxiter=1000, init=None)

        # Store fpc values for later
        fpcs3.append(fpc3)

        cntr, u, u0, d, jm, p, fpc4 = fuzz.cluster.cmeans(
            node4, ncenters, 2, error=0.005, maxiter=1000, init=None)

        # Store fpc values for later
        fpcs4.append(fpc4)

    plt.show()

    plt.plot(x, fpcs1, label="Nó 1")
    plt.plot(x, fpcs2, label="Nó 2")
    plt.plot(x, fpcs3, label="Nó 3")
    plt.plot(x, fpcs4, label="Nó 4")
    plt.xlabel("Número de Centros")
    plt.ylabel("FCP")
    plt.tight_layout()
    plt.legend()
    plt.grid()
    fig = plt.gcf()
    plt.show()
    plt.draw()
    fig.savefig("img/fpc_nCluster.png")