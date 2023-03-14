import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QPixmap

class MyWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Load image file
        image_file = 'path/to/image.jpg'
        pixmap = QPixmap(image_file)

        # Create label widget and set pixmap
        self.label = QLabel(self)
        self.label.setPixmap(pixmap)
        self.label.setGeometry(50, 50, pixmap.width(), pixmap.height())

        # Set window properties
        self.setWindowTitle('My Application')
        self.setGeometry(100, 100, pixmap.width() + 100, pixmap.height() + 100)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = MyWidget()
    widget.show()
    sys.exit(app.exec_())
