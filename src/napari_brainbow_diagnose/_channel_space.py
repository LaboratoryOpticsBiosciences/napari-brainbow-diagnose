from qtpy.QtWidgets import QSpacerItem, QSizePolicy
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QSpinBox,
)
from qtpy.QtCore import Qt
from superqt import QDoubleRangeSlider

import pyqtgraph as pg
import numpy as np
import napari

class channel_space_widget(QWidget):
    """
    Show the pixel in some color space
    """

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # This container will contain the RGB plot
        graph_container = QWidget()

        # RGB space view
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.graphics_widget.setBackground(None)
#        graph_container.setMaximumHeight(100)
        graph_container.setLayout(QHBoxLayout())
        graph_container.layout().addWidget(self.graphics_widget)
        
        
        # populate plot
        btn_pop = QPushButton("Populate")
        btn_pop.clicked.connect(self._populate)

        self.setLayout(QVBoxLayout())

        self.layout().addWidget(graph_container)
        self.layout().addWidget(btn_pop)



    def selected_image_layers(self):
        selected_layers= [
            layer
            for layer in self.viewer.layers.selection
            if isinstance(layer, napari.layers.Image)
        ]
        
        if len(selected_layers)!=3:
            raise ValueError('Exaclty 3 layers have to be selected')
            
        return selected_layers

    def _populate(self):
       
        self.redraw()


    def redraw(self, rebuild_gui=True):
        # add a new plot to the graphics_widget or empty the old plot
        if not hasattr(self, "plot"):
            self.plot = self.graphics_widget.addPlot()
        else:
            self.plot.clear()
        xyz=[]
        for layer in self.selected_image_layers():
            xyz.append(layer.data[:])
        xyz=np.array(xyz).reshape((3,-1)).transpose().astype('float')
        rtp=toSpherical_np(xyz[::10,:])        
        
        self.plot.plot(rtp[:,1],rtp[:,2],pen=None,symbol='o',symbolSize=0.1,alpha=0.01)
        
#https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
def toSpherical_np(xyz):
        ptsnew = np.zeros(xyz.shape)
        xy = xyz[:,0]**2 + xyz[:,1]**2
        ptsnew[:,0] = np.sqrt(xy + xyz[:,2]**2)
        #ptsnew[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        ptsnew[:,1] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        ptsnew[:,2] = np.arctan2(xyz[:,1], xyz[:,0])
        return ptsnew        
        
#@napari_hook_implementation
#def napari_experimental_provide_dock_widget():
#    # you can return either a single widget, or a sequence of widgets
#    return [channel_space_widget]

