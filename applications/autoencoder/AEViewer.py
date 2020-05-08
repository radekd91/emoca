import os, sys
mv_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", "..", ".."))
if mv_path not in sys.path:
    sys.path += [mv_path]

from MeshVista.VisualizationWidgets import PyVistaWindow
from MeshVista.Visualization import PyVistaVis, VisType
from MeshVista.Mesh import VistaMesh

from AEMeshGenerator import AEMeshGenerator

from PyQt5 import Qt
from PyQt5.QtCore import pyqtSlot, pyqtSignal
import traceback
import numpy as np

class AEViewer(PyVistaWindow):
    render_trigger = pyqtSignal()

    def __init__(self,
                 generator: [AEMeshGenerator, list],
                 parent=None,
                 show=True):
        # 1) initialize app logic related members
        if isinstance(generator, list):
            self.generator = generator[0]
            self.generator_list = generator
        elif isinstance(generator, AEMeshGenerator):
            self.generator = generator
            self.generator_list = [generator]
        else:
            raise ValueError("Invalid designer parameter")

        # 2) set up the window
        shape = (1, 1 + len(self.generator_list))
        super().__init__(parent, shape=shape)
        self.setAcceptDrops(True)
        self.setWindowTitle("Autoencoder Viewer")

        self.generator_visualizations = []
        self.generator_latent_space_widgets = []
        for vi, vis in enumerate(self.generator.get_visualizations()):
            vis.subplot_index = (0, vi+1)
            self.add_visualization(vis)
            self.vtk_widget.add_text("Decoded", font_size=10)
            self.generator_visualizations += [vis]
        self.vtk_widget.link_views()

        num_visualizations_per_generator = len(self.generator.get_visualizations())
        for i in range(num_visualizations_per_generator):
            visualizations = [generator.get_visualizations()[i] for generator in self.generator_list]
            # self.add_visualization(visualizations)
            widget = self.vis_widget.addWidget(visualizations)
            self.generator_latent_space_widgets += [widget]

        self.input_mesh = None
        self.output_mesh = None
        self.input_mesh_vis = None

        # 3) create menu
        main_menu = self.menuBar()
        file_menu = main_menu.addMenu('File')
        save_action = Qt.QAction('Save', self)
        save_action.triggered.connect(self.save)
        save_action.setShortcut('Ctrl+S')

        open_action = Qt.QAction('Open', self)
        open_action.triggered.connect(self.open)
        open_action.setShortcut('Ctrl+O')
        exit_button = Qt.QAction('Exit', self)
        exit_button.triggered.connect(self.close)
        exit_button.setShortcut('Ctrl+Q')

        file_menu.addAction(save_action)
        file_menu.addAction(open_action)
        file_menu.addAction(exit_button)

        edit_menu = main_menu.addMenu('Edit')
        reset_action = Qt.QAction('Reset', self)
        reset_action.triggered.connect(self.reset)
        reset_action.setShortcut('Ctrl+R')
        edit_menu.addAction(reset_action)

        self.start_rendering(show)

    def reset(self):
        try:
            if self.input_mesh is not None:
                self.encode_input_mesh()
        except Exception as e:
            print(traceback.print_exc())

    def save(self):
        filename = self._get_save_filename_though_dialog("Save output mesh")
        self.output_mesh.export_surface(filename)
        print("Mesh saved to '%s'" % filename)

    def open(self):
        filename = self._get_open_filename_though_dialog("Load input mesh")
        print("Opening mesh from '%s'" % filename)
        self.open_mesh(filename)

    def open_mesh(self, filename):
        try:
            self.input_mesh = VistaMesh(path=filename)

            if self.input_mesh_vis is not None:
                self.input_mesh_vis.set_points(self.input_mesh.get_vertices())
            else:
                self.input_mesh_vis = PyVistaVis(self.input_mesh.pyvista_surface(), "Input Mesh", VisType.SURFACE, subplot_index=(0, 0))
                self.add_visualization(self.input_mesh_vis)
                self.vtk_widget.add_text("Input Mesh", font_size=10)
                self.vis_widget.addWidget(self.input_mesh_vis)

            self.encode_input_mesh()

        except Exception as e:
            print(traceback.print_exc())

    def encode_input_mesh(self):
        verts = self.input_mesh.get_vertices()
        for gi, gen in enumerate(self.generator_list):
            z = gen.encode_mesh(verts)
            # z, output_mesh = gen.encode_mesh(verts)
            self.generator_latent_space_widgets[gi].setValue("latent_code", z)

            self.output_mesh = gen.get_current_mesh()
            self.generator_visualizations[gi].set_scalar_field(
                np.linalg.norm(self.input_mesh.get_vertices() - self.output_mesh.get_vertices(), axis=1))

            print("Mesh encoded to:")
            print(z)

    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat('text/uri-list'):
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        # filename = e.mimeData().text()[8:]
        filename = e.mimeData().urls()[0].toLocalFile()
        print(filename)
        self.open_mesh(filename)


if __name__ == "__main__":
    app = Qt.QApplication(sys.argv)
    generator = AEMeshGenerator("results/COMA/2020_05_05_23_51_24_Coma")
    window = AEViewer(generator)
    sys.exit(app.exec_())

