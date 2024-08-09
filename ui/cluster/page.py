

# UI Libraries
from PySide6.QtCore import Qt, SignalInstance
from PySide6.QtWidgets import QVBoxLayout, QLabel, QHBoxLayout, QGridLayout
import sys
import plotly.graph_objects as go
import plotly.express as px
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtWebEngineWidgets import QWebEngineView

# Custom UI libraries
from ui.components.page import Page

#matplotlib integration libraries
import sys
import time

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure

import pandas as pd
import pickle
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

import mplcursors


class ClusterPage(Page):
    def __init__(
        self,
        inputFile: SignalInstance,
        outputDir: SignalInstance,
        onTabSelected: SignalInstance,
    ) -> None:
        super().__init__(inputFile, outputDir, onTabSelected)
        self._layout = QGridLayout()
        self._layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.setLayout(self._layout)

        

        self.X = pd.read_csv('data/Full_X.csv')
        self.X.drop(columns=self.X.columns[0], axis=1, inplace=True)
        minimax = MinMaxScaler()
        minimax.fit_transform(self.X)
        self.labels = pd.read_csv('data/Y_yea.csv', header=None).to_numpy()[:, 1]

        self.clusterer = OPTICS(min_samples = 0.1)
        self.clusters = self.clusterer.fit_predict(self.X)
        self.clusters_s = np.array([str(i) for i in self.clusters])

        classifiers = []
        self.rf = pickle.load(open('data/rf.pkl', 'rb'))
        # fig = self.genMesh("ENSG00000260592", "ENSG00000239998") 
        fig = self.dataDR()
        canvas = FigureCanvas(fig)
        self._layout.addWidget(NavigationToolbar(canvas, self), 0, 0)
        self._layout.addWidget(canvas, 1, 0)

        
        plot_html = self.parallel_coords_plot()
        self.browser = QWebEngineView()
        self._layout.addWidget(self.browser, 1, 1, alignment=Qt.AlignmentFlag.AlignTop)
        self.browser.setHtml(plot_html)
        self._layout.setColumnStretch(1, 1)
        self._layout.setColumnStretch(0, 0)
        


    def parallel_coords_plot(self):
        
        # self.X['cluster'] = self.clusters
        # fig = px.parallel_coordinates(self.X, color='cluster')
        # self.X.drop(columns='cluster', axis=1, inplace=True)
        # plot_html = fig.to_html(include_plotlyjs='cdn')




        xt = self.X.T
        xt['Marker'] = xt.index
        xt = pd.melt(xt, id_vars=['Marker'])
        xt['Cluster'] = self.clusters_s[list(xt['variable'])]
        fig = px.scatter(xt, x=xt.Marker, y=xt.value, color=xt.Cluster)
        plot_html = fig.to_html(include_plotlyjs='cdn')
        

    




        return plot_html
        
    def dataDR(self):
        preds = self.rf.predict_proba(self.X)[:, 1]
        preds = np.array([pred > 0.5 for pred in preds])

        correct_preds = [i for i in range(len(preds)) if preds[i] == self.labels[i]]
        incorrect_preds = [i for i in range(len(preds)) if preds[i] != self.labels[i]]

        fig = plt.figure()
        
        x_low = pd.read_csv('data/X_Low_Final.csv', header=None).to_numpy()
        ax = fig.add_subplot(3, 1, 1, projection='3d')
        graph = ax.scatter(x_low[:, 0], x_low[:, 1], x_low[:, 2], c=self.labels, cmap=sns.diverging_palette(250, 15, center="dark", as_cmap=True), alpha=1.0)
        ax.set_title("Labeled Data")

        ax2 = fig.add_subplot(3, 1, 2, projection='3d', sharex=ax, sharey=ax, sharez=ax)
        ax2.set_title("Predictions")
        # graph = ax2.scatter(x_low[correct_preds, 0], x_low[correct_preds, 1], x_low[correct_preds, 2], c=preds[correct_preds], cmap=sns.diverging_palette(250, 15, center="dark", as_cmap=True), alpha=1.0, marker='o')
        graph = ax2.scatter(x_low[incorrect_preds, 0], x_low[incorrect_preds, 1], x_low[incorrect_preds, 2], c=preds[incorrect_preds], cmap=sns.diverging_palette(250, 15, center="dark", as_cmap=True), alpha=1.0, marker='x', s=100)

        graph = ax2.scatter(x_low[:, 0], x_low[:, 1], x_low[:, 2], c=preds, cmap=sns.diverging_palette(250, 15, center="dark", as_cmap=True), alpha=1.0, marker='o')
        ax.shareview(ax2)

        ax3 = fig.add_subplot(3, 1, 3, projection='3d', sharex=ax, sharey=ax, sharez=ax)
        ax3.set_title("Clusters")
        scat = ax3.scatter(x_low[:, 0], x_low[:, 1], x_low[:, 2], c=self.clusters, cmap=cm.tab10, alpha=1.0)
        h, l = scat.legend_elements()
        # ax3.legend(handles=h, labels = l)
        ax3.legend(handles=h, labels = l, bbox_to_anchor=(1.05, 1.0), loc='upper left')


        ax2.shareview(ax3)
        plt.tight_layout()

        
        return fig

    def genMesh(self, xvar=None, yvar=None):
        if not xvar: 
            xvar = self.X.columns[0]
            yvar = self.X.columns[1]
        # V1: Use median over all other variables
        new_samples = []
        medians = self.X.median()
        # medians.to_csv(os.path.join(output_dir, f"{stem}_medians.csv"))
        xvar_range = np.linspace(self.X[xvar].min(), self.X[xvar].max(), 30)
        yvar_range = np.linspace(self.X[yvar].min(), self.X[yvar].max(), 30)

        for val1 in xvar_range:
            for val2 in yvar_range: 
                sample = medians.copy()
                sample[xvar] = val1
                sample[yvar] = val2
                new_samples.append(sample)
        new_df = pd.DataFrame(new_samples)
        fig = plt.figure(figsize=(20, 15))
        fig.set_constrained_layout(True)
        ax = fig.add_subplot(projection='3d')
        ax.invert_xaxis()
        ax.scatter(new_df[xvar], new_df[yvar], self.rf.predict_proba(new_df)[:, 1], c=self.rf.predict_proba(new_df)[:, 1], cmap=cm.viridis)
        ax.set_xlabel(xvar, fontsize=10)
        ax.set_ylabel(yvar, fontsize=10)
        ax.set_zlabel("Probability", fontsize=10)
        ax.tick_params(labelsize='medium')
        ax.plot_trisurf(new_df[xvar], new_df[yvar], self.rf.predict_proba(new_df)[:, 1], cmap=cm.viridis)
        fig.suptitle("Classification Probabilities Across ENSG00000260592 and ENSG00000239998", y=0.95)
        return fig
    