import sys
import plotly.graph_objects as go
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtWebEngineWidgets import QWebEngineView

# Step 1: Create a Plotly plot
fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
plot_html = fig.to_html(include_plotlyjs='cdn')

# Step 2: Set up the Qt application
class PlotlyViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Plotly in PySide6')

        # Create a QWebEngineView widget
        self.browser = QWebEngineView()
        self.setCentralWidget(self.browser)

        # Load the Plotly HTML
        self.browser.setHtml(plot_html)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = PlotlyViewer()
    viewer.show()
    sys.exit(app.exec())
