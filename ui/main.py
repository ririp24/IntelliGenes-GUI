import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QTabWidget
from PySide6.QtCore import Signal
from pandas import DataFrame

from ui.components.page import Page

# Custom Components
from .input.page import InputPage
from .files.page import OutputFilesPage
from .pipeline.page import PipelinePage


class MainWindow(QMainWindow):
    # global state for input and output file
    # will be either a valid path or an empty string
    inputFile = Signal(str)
    outputDir = Signal(str)
    # Signals for when each page is selected
    inputPageSignal = Signal()
    pipelinePageSignal = Signal()
    filesPageSignal = Signal()


    def __init__(self):
        super().__init__()
        self.setWindowTitle("IntelliGenes")

        layout = QVBoxLayout()

        tabs: list[tuple[str, Page]] = [
            ("Input", InputPage(self.inputFile, self.outputDir, self.inputPageSignal)),
            ("Pipeline", PipelinePage(self.inputFile, self.outputDir, self.pipelinePageSignal)),
            ("Files", OutputFilesPage(self.inputFile, self.outputDir, self.filesPageSignal)),
        ]
        def select_tab(index: int):
            tabs[index][1].onTabSelected.emit()
        
        self.inputFile.emit("")
        self.outputDir.emit("")

        tab_bar = QTabWidget()
        tab_bar.currentChanged.connect(select_tab)
        tab_bar.setTabPosition(QTabWidget.TabPosition.North)
        tab_bar.setDocumentMode(True)

        for name, widget in tabs:
            tab_bar.addTab(widget, name)
        tab_bar.setCurrentIndex(0)
        

        tab_bar.setLayout(layout)

        self.setCentralWidget(tab_bar)


def run():
    app = QApplication([])

    window = MainWindow()
    window.show()

    sys.exit(app.exec())