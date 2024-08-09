

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
import seaborn as sns

import mplcursors


class VizPage(Page):
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
        self.labels = pd.read_csv('data/Y_yea.csv', header=None).to_numpy()[:, 1]

        self.clusterer = OPTICS(min_samples = 0.1)
        self.clusters = self.clusterer.fit_predict(self.X)

        classifiers = []
        self.rf = pickle.load(open('data/rf.pkl', 'rb'))

        fig = self.genMesh(xvar='ENSG00000241553', yvar='ENSG00000233276')
        canvas = FigureCanvas(fig)
        self._layout.addWidget(NavigationToolbar(canvas, self), 0, 0)
        self._layout.addWidget(canvas, 1, 0)
        


    def genMesh(self, xvar=None, yvar=None):
        if not xvar: 
            xvar = self.X.columns[0]
        if not yvar:
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
        fig.suptitle(f"Classification Probabilities Across {xvar} and {yvar}", y=0.95)
        return fig
    