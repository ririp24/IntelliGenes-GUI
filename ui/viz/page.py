

# UI Libraries
from PySide6.QtCore import Qt, SignalInstance
from PySide6.QtWidgets import QVBoxLayout, QLabel

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

import mplcursors


class VizPage(Page):
    def __init__(
        self,
        inputFile: SignalInstance,
        outputDir: SignalInstance,
        onTabSelected: SignalInstance,
    ) -> None:
        super().__init__(inputFile, outputDir, onTabSelected)
        self._layout = QVBoxLayout()
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setLayout(self._layout)

        self.X = pd.read_csv('data/Full_X.csv')
        self.X.drop(columns=self.X.columns[0], axis=1, inplace=True)
        self.labels = pd.read_csv('data/Y_yea.csv', header=None).to_numpy()[:, 1]

        classifiers = []
        self.rf = pickle.load(open('data/rf.pkl', 'rb'))
        # fig = self.genMesh("ENSG00000260592", "ENSG00000239998") 
        fig = self.dataDR()
        canvas = FigureCanvas(fig)
        self._layout.addWidget(NavigationToolbar(canvas, self))
        self._layout.addWidget(canvas)
        



        
    def dataDR(self):
        preds = self.rf.predict_proba(self.X)[:, 1]
        preds = [pred > 0.5 for pred in preds]


        fig = plt.figure()
        
        x_low = pd.read_csv('data/X_Low_Final.csv', header=None).to_numpy()
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        graph = ax.scatter(x_low[:, 0], x_low[:, 1], x_low[:, 2], c=self.labels, cmap=cm.bwr, alpha=1.0)
        ax.set_title("Labeled Data")

        ax2 = fig.add_subplot(1, 2, 2, projection='3d', sharex=ax, sharey=ax, sharez=ax)
        ax2.set_title("Predictions")
        graph = ax2.scatter(x_low[:, 0], x_low[:, 1], x_low[:, 2], c=preds, cmap=cm.bwr, alpha=1.0)

        ax.shareview(ax2)

        mplcursors.cursor(fig)
        
        return fig

    def genMesh(self, xvar=None, yvar=None):
        if not xvar: 
            xvar = self.X.columns[0]
            yvar = self.X.columns[1]
        # V1: Use median over all other variables
        new_samples = []
        medians = self.X.median()
        # medians.to_csv(os.path.join(output_dir, f"{stem}_medians.csv"))
        xvar_range = np.linspace(self.X[xvar].min(), self.X[yvar].max(), 30)
        yvar_range = np.linspace(self.X[xvar].min(), self.X[yvar].max(), 30)

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
    