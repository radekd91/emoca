import os, sys
mv_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", "..", ".."))
if mv_path not in sys.path:
    sys.path += [mv_path]

from MeshVista.VisualizationWidgets import PyVistaWindow

from AEMeshGenerator import AEMeshGenerator

from PyQt5 import Qt
from PyQt5.QtCore import pyqtSlot, pyqtSignal


class AEViewer(PyVistaWindow):
    render_trigger = pyqtSignal()

    def __init__(self,
                 generator: [AEMeshGenerator, list],
                 parent=None,
                 show=True):
        super().__init__(parent)

        self.setWindowTitle("Implant Designer")

        # 2) initialize app logic related members
        if isinstance(generator, list):
            self.generator = generator[0]
            self.generator_list = generator
        elif isinstance(generator, AEMeshGenerator):
            self.generator = generator
            self.generator_list = [generator]
        else:
            raise ValueError("Invalid designer parameter")

        for vis in self.generator.get_visualizations():
            self.add_visualization(vis)

        num_visualizations_per_generator = len(self.generator.get_visualizations())
        for i in range(num_visualizations_per_generator):
            visualizations = [generator.get_visualizations()[i] for generator in self.generator_list]
            # self.add_visualization(visualizations)
            self.vis_widget.addWidget(visualizations)



        # 3) create menu
        main_menu = self.menuBar()
        file_menu = main_menu.addMenu('File')
        save_action = Qt.QAction('Save', self)
        open_action = Qt.QAction('Open', self)
        exit_button = Qt.QAction('Exit', self)
        exit_button.setShortcut('Ctrl+Q')

        self.start_rendering(show)



if __name__ == "__main__":
    app = Qt.QApplication(sys.argv)
    generator = AEMeshGenerator("results/COMA/2020_05_05_23_51_24_Coma")
    window = AEViewer(generator)

    sys.exit(app.exec_())

