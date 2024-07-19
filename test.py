import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import pickle
import pacmap
# import umap
# import umap.plot
import json
import re, seaborn as sns
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import mplcursors



import matplotlib.pyplot as plt

def main():

    classifiers = []
    rf = pickle.load(open('data/rf.pkl', 'rb'))
    X = pd.read_csv('data/Full_X.csv')
    X.drop(columns=X.columns[0], axis=1, inplace=True)
    
    labels = pd.read_csv('data/Y_yea.csv', header=None).to_numpy()[:, 1]




    # x_low = np.loadtxt('data/X_Low.csv', delimiter=',')

    def genMesh(xvar=X.columns[0], yvar= X.columns[1]):

        # V1: Use median over all other variables
        new_samples = []
        medians = X.median()
        # medians.to_csv(os.path.join(output_dir, f"{stem}_medians.csv"))
        xvar_range = np.linspace(X[xvar].min(), X[yvar].max(), 30)
        yvar_range = np.linspace(X[xvar].min(), X[yvar].max(), 30)

        for val1 in xvar_range:
            for val2 in yvar_range:
                sample = medians.copy()
                sample[xvar] = val1
                sample[yvar] = val2
                new_samples.append(sample)
        new_df = pd.DataFrame(new_samples)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(new_df[xvar], new_df[yvar], rf.predict_proba(new_df)[:, 1], c=rf.predict_proba(new_df)[:, 1], cmap=cm.viridis)
        ax.set_xlabel(xvar, fontsize=10)
        ax.set_ylabel(yvar, fontsize=10)
        ax.set_zlabel("Probability", fontsize=10)
        ax.tick_params(labelsize='medium')
        ax.plot_trisurf(new_df[xvar], new_df[yvar], rf.predict_proba(new_df)[:, 1], cmap=cm.viridis)
        plt.show()
        # fig.set_size_inches(2 + num_selected_features * 0.8, 2 + num_selected_features * 0.8)
        # set_fig_labels(fig, title=("testMesh"))
        # save_fig(fig, os.path.join(output_dir, f"{stem}_Test_Mesh.png"))
        return
    
    def make_PacMAP():
        # matrix = x_t.to_numpy()
        
        pass

    def test_PaCMAP():
        """testing
        X = np.load("data/coil_20.npy", allow_pickle=True)
        X = X.reshape(X.shape[0], -1)
        y = np.load("data/coil_20_labels.npy", allow_pickle = True)

        transformer = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0)


        data = np.random.rand(1000, 10)
        embed_pacmap = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0).fit_transform(data)
        print("1")
        transformer.fit(data)
        print("2")
        X_low = transformer.transform(data, init="pca")"""

        preds = rf.predict_proba(X)[:, 1]
        preds = [pred > 0.5 for pred in preds]


        fig = plt.figure()
        
        x_low = pd.read_csv('data/X_Low_Final.csv', header=None).to_numpy()
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        graph = ax.scatter(x_low[:, 0], x_low[:, 1], x_low[:, 2], c=labels, cmap=cm.bwr, alpha=1.0)
        ax.set_title("Labeled Data")

        ax2 = fig.add_subplot(1, 2, 2, projection='3d', sharex=ax, sharey=ax, sharez=ax)
        ax2.set_title("Predictions")
        graph = ax2.scatter(x_low[:, 0], x_low[:, 1], x_low[:, 2], c=preds, cmap=cm.bwr, alpha=1.0)

        ax.shareview(ax2)
        
        plt.show()

        # mplcursors.cursor(graph)

        return

    def test_UMAP():
        data = np.random.rand(1000, 10)
        transformer = umap.UMAP(n_neighbors = 10, min_dist = 0.1, metric = 'euclidean', n_components = 2, random_state=42, n_jobs=1)
        X_low = transformer.fit_transform(data)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # plt.scatter(X_low[:, 0], X_low[:, 1], X_low[:, 3])
        # # plt.gca().set_aspect('equal', 'datalim')
        # plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
        # plt.title('UMAP projection of the Digits dataset', fontsize=24)
        plt.show()

    def make_UMAP():
        # matrix = x_t.to_numpy()
        pass



    # genMesh("ENSG00000260592", "ENSG00000239998")
    # make_PacMAP()
    test_PaCMAP()


   


    # x = np.linspace(0, 2*np.pi, 100)
    # y = np.sin(3*x)
    # ax.plot(x, y)

    # plt.show()

if __name__=="__main__": 
    main() 