import napari

import pandas as pd
import numpy as np

from magicgui import magicgui
from magicgui import magic_factory
from ._utils_channel_space import rgb_to_maxwell_triangle
import pathlib

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
    df['maxwell_x'], df['maxwell_y'] = rgb_to_maxwell_triangle(df['red'], df['green'], df['blue'])
    features_table = {'red': df['red'].values, 'green': df['green'].values, 'blue': df['blue'].values, 'maxwell_x': df['maxwell_x'].values, 'maxwell_y': df['maxwell_y'].values}
    points_layer.features = features_table