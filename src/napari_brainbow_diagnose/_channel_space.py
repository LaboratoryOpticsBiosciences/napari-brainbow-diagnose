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
        
        # individual layers: min/max sliders
        self.sliders = QWidget()
        self.sliders.setLayout(QVBoxLayout())
        self.sliders.layout().setSpacing(0)


        
        # populate plot
        btn_pop = QPushButton("Populate")
        btn_pop.clicked.connect(self._populate)



        self.setLayout(QVBoxLayout())

        self.layout().addWidget(graph_container)
        self.layout().addWidget(self.sliders)        
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


    def redraw(self, rebuild_gui=True,rlim=None):
        # add a new plot to the graphics_widget or empty the old plot
        if not hasattr(self, "plot"):
            self.plot = self.graphics_widget.addPlot()
        else:
            self.plot.clear()
                   
        
        #find colors            
        colors = []
        for layer in self.selected_image_layers():
            colormap = layer.colormap.colors
            color = np.asarray(colormap[-1, 0:3]) * 255
            colors.append(color)
            
            
            
            
        xyz=[]
        cur=self.viewer.dims.point
        for layer in self.selected_image_layers():
            xyz.append(np.clip(layer.data[int(cur[0]),:,:],layer.contrast_limits[0],layer.contrast_limits[1]))
        xyz=np.array(xyz).reshape((3,-1)).transpose().astype('float')
        rtp=toSpherical_np(xyz[:,:])  
        
        rmax=rtp[:,0].max()
        
        if rlim !=None:
            rtp=rtp[rtp[:,0]>rlim[0],:]
            rtp=rtp[rtp[:,0]<rlim[1],:]

        if rlim ==None:
            rlim=[rtp[:,0].min(),rtp[:,0].max()]        
        self.plot.plot(rtp[:,1],rtp[:,2],pen=None,symbol='o',symbolSize=0.1,alpha=0.01)

        # update sliders
        if rebuild_gui:
            layout = self.sliders.layout()
            for i in reversed(range(layout.count())):
                layout.itemAt(i).widget().setParent(None)
            row = RadiusRGBLimitsWidget(rmax,rlim, self
                )

            layout.addWidget(row)
        
        
#from https://github.com/haesleinhuepf/napari-brightness-contrast  
class RadiusRGBLimitsWidget(QWidget):
    """
    This widget corresponds to a single line represeting a layer with the option to configure min/max contrast limits.
    """

    def __init__(self,rmax,rlim, gui):
        super().__init__(gui)

        self.setLayout(QHBoxLayout())

        lbl = QLabel('Radius')
        #lbl.setStyleSheet("color: #%02x%02x%02x" % tuple(color.astype(int)))
        self.layout().addWidget(lbl)

        # show min/max intensity
        lbl_min = QLabel()
        lbl_min.setText("{:.2f}".format(0))
        lbl_max = QLabel()
        lbl_max.setText("{:.2f}".format(rmax))

        # allow to tune min and max within one slider
        slider = QDoubleRangeSlider()
        slider.setOrientation(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(rmax)
        slider.setValue(rlim)
        slider.setSingleStep(rmax / 1000)

        # update on change
        def value_changed(): 
            lbl_min.setText("{:.2f}".format(slider.value()[0]))
            lbl_max.setText("{:.2f}".format(slider.value()[1]))
            gui.redraw(rlim=slider.value(),rebuild_gui=False)

        slider.valueChanged.connect(value_changed)
        
        self.layout().addWidget(lbl_min)
        self.layout().addWidget(slider)
        self.layout().addWidget(lbl_max)
        
    
#https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
def toSpherical_np(xyz):
        ptsnew = np.zeros(xyz.shape)
        xy = xyz[:,0]**2 + xyz[:,1]**2
        ptsnew[:,0] = np.sqrt(xy + xyz[:,2]**2)
        #ptsnew[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        ptsnew[:,1] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        ptsnew[:,2] = np.arctan2(xyz[:,1], xyz[:,0])
        return ptsnew        
        
def min_max(data):
    if "dask" in str(type(data)):  # ugh
        data = np.asarray(data)

    return float(data.min()), float(data.max())
    
#@napari_hook_implementation
#def napari_experimental_provide_dock_widget():
#    # you can return either a single widget, or a sequence of widgets
#    return [channel_space_widget]

