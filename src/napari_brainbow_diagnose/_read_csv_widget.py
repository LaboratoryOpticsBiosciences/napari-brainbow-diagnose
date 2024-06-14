import napari
from napari.layers import Points

import pandas as pd
import numpy as np

from magicgui import magicgui
from magicgui import magic_factory
from ._utils_channel_space import rgb_to_maxwell_triangle, calculate_brightness
import pathlib

@magic_factory(call_button="(test) Create a Points layer", auto_call=False)
def read_csv_widget(filename = pathlib.Path('/path/to/csv/file.csv')):
    # Getting the current viewer
    viewer = napari.current_viewer()
    # Import data
    df = pd.read_csv(filename, usecols=range(6), names=['axis-0', 'axis-1', 'axis-2', 'red', 'green', 'blue'])
    # Create points layer
    coordinates = df[['axis-0', 'axis-1', 'axis-2']].values
    colors = df[['red', 'green', 'blue']].values
    points_layer = viewer.add_points(coordinates, name='points', face_color=colors, edge_color=colors, size=20, out_of_slice_display=True)
    
    # Add Maxwell coordinates
    df['maxwell_x'], df['maxwell_y'] = rgb_to_maxwell_triangle(df['red'], df['green'], df['blue'])
    df['brightness'] = calculate_brightness(df['red'], df['green'], df['blue'])

    df = df.sort_values(by='brightness').reset_index(drop=True)
    
    features_table = {
        'red': df['red'].values, 'green': df['green'].values, 'blue': df['blue'].values, 
        'maxwell_x': df['maxwell_x'].values, 'maxwell_y': df['maxwell_y'].values,
        'brightness_index': df.index, 'brightness': df['brightness']
        }
    points_layer.features = features_table


# Add color features to an existing points layer which has features 'red', 'green', and 'blue'
@magic_factory(call_button="Add Features to Points Layer", auto_call=False, points_layer={"label": "Select Points Layer"}, points_selection={"label": "Add selected points feature"})
def add_features_widget(points_layer: Points, points_selection: bool):
    # Extract existing features
    if not all(col in points_layer.features for col in ['red', 'green', 'blue']):
        raise ValueError("Selected points layer does not have 'red', 'green', and 'blue' features.")
    
    red = points_layer.features['red']
    green = points_layer.features['green']
    blue = points_layer.features['blue']

    # Add Maxwell coordinates
    maxwell_x, maxwell_y = rgb_to_maxwell_triangle(red, green, blue)
    brightness = calculate_brightness(red, green, blue)

    features_table = {
        'red': red.values,
        'green': green.values,
        'blue': blue.values,
        'maxwell_x': maxwell_x.values,
        'maxwell_y': maxwell_y.values,
        'brightness': brightness.values,
        'range': np.arange(len(brightness))
    }

    for key, value in features_table.items():
        points_layer.features[key] = value

    if points_selection:
        selected = points_layer.selected_data
        clustering = np.zeros(len(points_layer.data))
        for point_index in selected:
            clustering[point_index] = 1

        points_layer.features['CLUSTER'] = clustering.astype(int)
