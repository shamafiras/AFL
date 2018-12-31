from PySide2.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget, QSlider, QLabel, QApplication, QCheckBox
from PySide2 import QtGui, QtCore
from skimage import io
from src.transformer import Transformer


class BaseWidget(QWidget):
    def __init__(self, app_name='Base Widget'):
        super(BaseWidget, self).__init__()
        self.app_name = app_name
        # main window
        self.setGeometry(50, 50, 200, 200)
        self.setWindowTitle(app_name)
        # enable dragging and dropping onto the GUI
        self.setAcceptDrops(True)
        self.transformer = Transformer()

        ## make main layout
        #self.main_layout = QHBoxLayout()
        #self.generated_image_label = QLabel(self)
        #generated_image_layout = self.make_image_layout(self.generated_image_label, 'Generated Images')
        #self.generated_image_label.setAlignment(QtCore.Qt.AlignHCenter)
        #self.main_layout.addLayout(generated_image_layout)
        #layout = QVBoxLayout()
        #layout.addLayout(self.main_layout)
        #self.setLayout(layout)
        #self.show()

    def make_image_layout(self, image_ql, title):
        txt = QLabel(title)
        image_layout = QVBoxLayout()
        image_layout.addWidget(txt)
        image_layout.addWidget(image_ql)
        txt.setAlignment(QtCore.Qt.AlignHCenter)
        return image_layout

    def make_slider_layout(self, value_changed_func, val=0):
        slider = QSlider(QtCore.Qt.Horizontal, self)
        slider.setValue(val)
        slider.valueChanged[int].connect(value_changed_func)
        layout_slider = QVBoxLayout()
        layout_slider.addWidget(slider)
        txt = QLabel()
        layout_slider.addWidget(txt)
        txt.setAlignment(QtCore.Qt.AlignHCenter)
        return layout_slider, slider, txt

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        if e.mimeData().hasUrls:
            e.setDropAction(QtCore.Qt.CopyAction)
            e.accept()
            # Workaround for OSx dragging and dropping
            for url in e.mimeData().urls():
                file_name = str(url.toLocalFile())
            self.drop_event_file_name = file_name
            self.execute_drop_event(file_name)
        else:
            e.ignore()

    def execute_drop_event(self, file_name):
        pass

    def load_image(self):
        self.loaded_image = io.imread(self.file_name)


if __name__ == '__main__':
    # Initialise the application
    app = QApplication([])
    # Call the main widget
    ex = BaseWidget()
    app.exec_()
