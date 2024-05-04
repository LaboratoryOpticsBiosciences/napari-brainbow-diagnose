import napari

import pandas as pd
import numpy as np

from magicgui import magicgui
from magicgui import magic_factory
import pathlib

def rgb_to_xy(r, g, b):
    s = r+g+b
    r = r/s
    g = g/s
    b = b/s
    x = (r-b)/np.sqrt(3)
    y = g
    return x, y

@magic_factory(call_button="Create a Points layer", auto_call=False)
def read_csv_widget(filename = pathlib.Path('/path/to/csv/file.csv')):
    # Getting the current viewer
    viewer = napari.current_viewer()
    # Import data
    df = pd.read_csv(filename, names=['axis-0', 'axis-1', 'axis-2', 'red', 'green', 'blue'])
    # Create points layer
    coordinates = df[['axis-0', 'axis-1', 'axis-2']].values
    colors = df[['red', 'green', 'blue']].values
    points_layer = viewer.add_points(coordinates, name='points', face_color=colors, edge_color=colors, size=20, out_of_slice_display=True)
    
    # Add Maxwell coordinates
    df['x'], df['y'] = rgb_to_xy(df['red'], df['green'], df['blue'])
    features_table = {'red': df['red'].values, 'green': df['green'].values, 'blue': df['blue'].values, 'x': df['x'].values, 'y': df['y'].values}
    points_layer.features = features_table