"""
Utilities module
"""

# Copyright (C) 2021  Blue Brain Project, EPFL
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import configparser
import random

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# from layer_recognition.geometry import get_layer_thickness


def get_config(config_file_path: str):
    """
    read config file and return either existing entries or default values
    Args:
        config_file_path (str): The full path of the configuration path
    Returns:
        list of configuration entries
    """

    config = configparser.ConfigParser()
    config.sections()
    config.read(config_file_path)

    try:
        input_detection_directory = config["BATCH"]["input_detection_directory"]
        cell_position_suffix = config["BATCH"]["cell_position_suffix"].replace('"', "")
    except KeyError:
        input_detection_directory = None
        cell_position_suffix = None
    try:
        input_annotation_directory = config["BATCH"]["input_annotation_directory"]
        annotations_geojson_suffix = config["BATCH"]["annotations_geojson_suffix"]
    except KeyError:
        input_annotation_directory = None
        annotations_geojson_suffix = None

    try:
        exclude_flag = config.getboolean("BATCH", "exclude")
    except configparser.NoOptionError:
        exclude_flag = True

    pixel_size = float(config["BATCH"]["pixel_size"])

    output_path = config["BATCH"]["output_directory"]

    return (
        input_detection_directory,
        cell_position_suffix,
        input_annotation_directory,
        annotations_geojson_suffix,
        exclude_flag,
        pixel_size,
        output_path,
    )


def stereology_exclusion(dataframe):
    """
    The classical optical dissector method that adds a virtual -z coordinate
    to cells to be able to exclude from one slice, the cells located on two slices boundaries.

    Args:
        dataframe (pandas.dataframe): dataframe thats contains the cells X/Y coordinates

    Returns:
        The input dataframe wityh the new exlude column
    """
    random.seed(0)
    data = dataframe[["Centroid X µm", "Centroid Y µm"]].values
    nbrs = NearestNeighbors(n_neighbors=5, algorithm="kd_tree").fit(data)
    dataframe["mean_diameter"] = 0.5 * (
        dataframe["Max diameter µm"] + dataframe["Min diameter µm"]
    )

    def exclude(sample, slice_thickness=50):
        sample["neighbors"] = nbrs.kneighbors(data, 6, return_distance=False)[
            sample.name, :
        ]  # sample.name = row index
        neighbor_mean = dataframe.iloc[sample["neighbors"]]["mean_diameter"].mean()
        sample["neighbor_mean"] = neighbor_mean
        sample["exclude_for_density"] = (
            random.uniform(0, slice_thickness) + neighbor_mean / 2 >= slice_thickness
        )
        return sample

    dataframe_with_exclude_flag = dataframe.apply(exclude, axis=1)
    return dataframe_with_exclude_flag


def get_image_to_exlude_list(df_image_to_exclude):
    """
    Get the list of exclude image from the input dataframe
    Args:
        df_image_to_exclude(pandas.Dataframe)
    Returns:
        list of images name (str) to exclude
    """

    df_image_to_exclude = df_image_to_exclude.dropna().reset_index(drop=True)

    # Step 2: Apply a function to each value
    def remove_space(image_name):
        return image_name.replace(" ", "")

    new_image_name_column = df_image_to_exclude["Image ID to exclude"].apply(
        remove_space
    )

    # Step 3: Assign the new values back to the column
    df_image_to_exclude["Image ID to exclude"] = new_image_name_column
    return list(df_image_to_exclude["Image ID to exclude"])


def get_image_id(feature_path, cell_position_file_prefix="Features_"):
    """
    :param feature_path:
    :param cell_position_file_prefix:
    :return:
        str: the image unique id
    """
    feature_str_length = len(cell_position_file_prefix)
    prefix_pos = feature_path.rfind(cell_position_file_prefix)
    image_id = None
    if prefix_pos > -1:
        feature_pos = feature_path.rfind(cell_position_file_prefix) + feature_str_length
        image_id = feature_path[feature_pos : feature_path.find(".csv")]
    return image_id


import re


def get_animal_by_image_id(metadata_path):
    """
    From a metadata dataframe, returns a dictionary where the key is the image_id and the value the corresponding animal
    :param metadata_path:
    :return:
        a dictionary where the key is the image_id and the value the corresponding animal
    """
    meta_df = pd.read_csv(metadata_path, index_col=0)
    analyse_df = meta_df[meta_df.Analyze == True]
    image_id_by_animal = {}
    project_image = analyse_df[["Project_ID", "Image_Name"]].values.tolist()
    for project, image in project_image:
        underscore_pos = [m.start() for m in re.finditer("_", project)]
        animal = project[underscore_pos[0] + 1 : underscore_pos[1]]
        if animal[0] == "0":
            animal = animal[1:]
        image_id_by_animal[image] = animal
    return image_id_by_animal


def get_animals_id_list(meta_df):
    """
    From a metadata dataframe, returns a set that contains all the existing animal ids.
    :param meta_df: Pandas Dataframe
    :return:
        a set that contains all the existing animal ids
    """
    animals = set()

    analyse_df = meta_df[meta_df.Analyze == True]
    image_id_by_animal = {}
    project_image = analyse_df[["Project_ID", "Image_Name"]].values.tolist()
    for project, image in project_image:
        underscore_pos = [m.start() for m in re.finditer("_", project)]
        animal = project[underscore_pos[0] + 1 : underscore_pos[1]]
        if animal[0] == "0":
            animal = animal[1:]
        animals.add(animal)
    return animals


def get_s1hl_corners(df_points):
    top_left = df_points[df_points.index == "top_left"].to_numpy()[0]
    top_right = df_points[df_points.index == "top_right"].to_numpy()[0]
    bottom_right = df_points[df_points.index == "bottom_right"].to_numpy()[0]
    bottom_left = df_points[df_points.index == "bottom_left"].to_numpy()[0]
    return top_left, top_right, bottom_right, bottom_left


def gauss(x, mu, sigma, A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

