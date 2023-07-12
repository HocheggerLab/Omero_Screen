import napari
from omero_gallery.omero_gui_utils import MyWidget



def main():
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(MyWidget(viewer), area='right')
    napari.run()

if __name__ == '__main__':
    main()
