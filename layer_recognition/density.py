"""
QuPath porcessing for rat somatosensory cortex Nissl data module
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

from collections import defaultdict
import alphashape
import glob
import numpy as np
import pandas as pd
from shapely.geometry.multipolygon import MultiPolygon
import shapely

from layer_recognition.geometry import (
    count_nb_cell_per_polygon,
    create_depth_polygons,
    create_grid,
    get_bigger_polygon,
    get_inside_points,
    find_n_closest_pairs
)
from layer_recognition.io import get_cells_coordinate

from layer_recognition.utilities import (
    get_image_to_exlude_list,
    get_image_id,
    get_animal_by_image_id,
)
from layer_recognition.visualisation import (
    plot_densities,
    plot_densities_by_layer,
    plot_layers,
    plot_split_polygons_and_cell_depth,
)

# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals


def compute_depth_density(
    image_name,
    cells_features_df,
    points_annotations_path,
    s1hl_path,
    output_path,
    thickness_cut=50,
    nb_row=10,
    nb_col=10,
    visualisation_flag=False,
    save_plot_flag=False,
):
    """
    compute the cell densities as function of brain depth.
    If the cells_features_df contains a 'RF_prediction' columns, then
    the layer bundaries for each layer is also computed in percentage
    of brain depth
    Args:
        image_name:(str)
        cells_features_df:(panda.Dataframe)
        points_annotations_path:(str)
        s1hl_path:(str)
        output_path:(str)
        thickness_cut:(int)
        nb_row:(int)
        nb_col:(int)
        visualisation_flag:(bool)
        save_plot_flag=False:(bool)
    Returns:
        A panda.Dataframe that contains the cell densities as function of brain depth
    """
    (
        cells_centroid_x,
        cells_centroid_y,
        excluded_cells_centroid_x,
        excluded_cells_centroid_y,
    ) = get_cells_coordinate(cells_features_df)

    # Create grid from annotation
    s1_coordinates_dataframe = pd.read_csv(s1hl_path, index_col=0)
    points_annotations_dataframe = pd.read_csv(points_annotations_path, index_col=0)
    s1_coordinates = s1_coordinates_dataframe.to_numpy()
    top_left = points_annotations_dataframe[
        points_annotations_dataframe.index == "top_left"
    ].to_numpy()[0]
    top_right = points_annotations_dataframe[
        points_annotations_dataframe.index == "top_right"
    ].to_numpy()[0]
    bottom_right = points_annotations_dataframe[
        points_annotations_dataframe.index == "bottom_right"
    ].to_numpy()[0]
    bottom_left = points_annotations_dataframe[
        points_annotations_dataframe.index == "bottom_left"
    ].to_numpy()[0]

    horizontal_lines, vertical_lines = create_grid(
        top_left, top_right, bottom_left, bottom_right, s1_coordinates, nb_row, nb_col
    )

    split_polygons = create_depth_polygons(s1_coordinates, horizontal_lines)
    print("INFO: Computes the cells densities as function of percentage depth")
    nb_cell_per_slide = count_nb_cell_per_polygon(
        cells_centroid_x, cells_centroid_y, split_polygons
    )

    depth_percentage, densities, nb_cells = compute_cell_density(
        nb_cell_per_slide, split_polygons, thickness_cut / 1e3
    )

    layer_boundaries_dataframe = None
    if "RF_prediction" in cells_features_df.columns:
        # Compute the layer bundaries for each layer as a percentage of brain area depth
        layers = np.unique(cells_features_df.RF_prediction)
        boundaries = []
        for layer_name_a, layer_name_b in zip(layers[:-1], layers[1:]):
            population_top = cells_features_df[
                cells_features_df.RF_prediction == layer_name_a
            ][["Centroid X µm", "Centroid Y µm"]].to_numpy()
            population_bottom = cells_features_df[
                cells_features_df.RF_prediction == layer_name_b
            ][["Centroid X µm", "Centroid Y µm"]].to_numpy()

            nb_of_neigbors = 20
            closest_pairs = find_n_closest_pairs(
                population_top, population_bottom, nb_of_neigbors
            )
            neigbours = []
            for a_idx, b_idx, dist in closest_pairs:
                neigbours.append(population_top[a_idx])
            neigbours = np.array(neigbours)
            nb_cell_per_slide = count_nb_cell_per_polygon(
                neigbours[:, 0], neigbours[:, 1], split_polygons
            )

            boundaries.append(np.argmax(nb_cell_per_slide) / nb_row * 100)

        layer_boundaries_dataframe = pd.DataFrame(
            {
                "image": [image_name] * (len(layers) - 1),
                "layers": layers[:-1],
                "boundaries": boundaries,
            }
        )

    if visualisation_flag or save_plot_flag:
        plot_split_polygons_and_cell_depth(
            split_polygons,
            s1_coordinates,
            cells_centroid_x,
            cells_centroid_y,
            excluded_cells_centroid_x,
            excluded_cells_centroid_y,
            vertical_lines=vertical_lines,
            horizontal_lines=None,
            output_path=output_path,
            image_name=image_name,
            visualisation_flag=visualisation_flag,
            save_plot_flag=save_plot_flag,
        )

        plot_densities(
            depth_percentage,
            densities,
            output_path=output_path,
            image_name=image_name,
            visualisation_flag=visualisation_flag,
            save_plot_flag=save_plot_flag,
        )

    total_used_cells = sum(nb_cells)
    total_detected_cells = len(cells_centroid_x)
    if total_used_cells < total_detected_cells - total_detected_cells / 50:
        print(
            f"ERROR {image_name} there are {len(cells_centroid_x) - total_used_cells } "
            f"cells outside the grid for a total of {len(cells_centroid_x)} cells."
        )
        print(
            f"ERROR {image_name} there are  {total_used_cells}/{len(cells_centroid_x)}  used cells"
        )
        return None, None

    densities_dataframe = pd.DataFrame(
        {
            "image": [image_name] * len(depth_percentage),
            "depth_percentage": depth_percentage,
            "densities": densities,
        }
    )

    return densities_dataframe, layer_boundaries_dataframe


def single_image_process_per_layer(
    image_name,
    cell_position_file_path,
    output_path,
    df_image_to_exclude=None,
    thickness_cut=50,
    visualisation_flag=False,
    save_plot_flag=False,
    alpha=0.001,
):
    """
    compute the cell densities as function of brain depth for a single image
    Args:
        image_name:(str)
        cell_position_file_path:(str)
        output_path:(str)
        df_image_to_exclude:(pandas.Dataframe)
        thickness_cut:(int)
        visualisation_flag:(bool)
        save_plot_flag=False:(bool)
        alpha:(float)
    Returns:
        A panda.Dataframe that contains the cell densities per layer
    """

    if df_image_to_exclude is not None:
        images_to_exlude = get_image_to_exlude_list(df_image_to_exclude)
        search_name = image_name.replace("Features_", "")
        if search_name.replace(" ", "") in images_to_exlude:
            print(f"ERROR {search_name} is present in the df_image_to_exclude dataset")
            return None

    cells_features_df = pd.read_csv(cell_position_file_path, index_col=0)
    assert "exclude_for_density" in cells_features_df.columns

    per_layer_dataframe = None
    assert "RF_prediction" in cells_features_df

    if "RF_prediction" in cells_features_df:
        print("INFO: RF_prediction in cells_features_df")
        layers = np.unique(cells_features_df.RF_prediction)
        layers_densities, cells_pos_list, polygons = densities_from_layers(
            cells_features_df, layers, thickness_cut, alpha=alpha
        )

        if visualisation_flag or save_plot_flag:
            plot_layers(
                cells_pos_list,
                polygons,
                image_name,
                alpha,
                output_path,
                visualisation_flag,
            )
            plot_densities_by_layer(
                layers, layers_densities, image_name, output_path, visualisation_flag
            )
        layers_densities.insert(0, image_name)
        layers = np.insert(layers, 0, "Image")
        per_layer_dataframe = pd.DataFrame([layers_densities], columns=layers)

    return per_layer_dataframe


def single_image_process_per_depth(
    image_name,
    cell_position_file_path,
    points_annotations_path,
    s1hl_path,
    output_path,
    df_image_to_exclude=None,
    thickness_cut=50,
    nb_row=10,
    nb_col=10,
    visualisation_flag=False,
    save_plot_flag=False,
):
    """
    compute the cell densities as function of brain depth for a single image
    Args:
        image_name:(str)
        cell_position_file_path:(str)
        points_annotations_path:(str)
        s1hl_path:(str)
        output_path:(str)
        df_image_to_exclude:(pandas.Dataframe)
        thickness_cut:(int)
        nb_row:(int)
        nb_col:(int)
        visualisation_flag:(bool)
        save_plot_flag=False:(bool)
        alpha:(float)
        compute_per_layer:(bool)
        compute_per_depth:(bool)
    Returns:
        A panda.Dataframe that contains the cell densities as function of brain depth
    """

    if df_image_to_exclude is not None:
        images_to_exlude = get_image_to_exlude_list(df_image_to_exclude)
        search_name = image_name.replace("Features_", "")
        if search_name.replace(" ", "") in images_to_exlude:
            print(f"ERROR {search_name} is present in the df_image_to_exclude dataset")
            return None, None

    cells_features_df = pd.read_csv(cell_position_file_path, index_col=0)
    assert "exclude_for_density" in cells_features_df.columns

    percentage_dataframe, layer_boundaries_dataframe = compute_depth_density(
        image_name,
        cells_features_df,
        points_annotations_path,
        s1hl_path,
        output_path,
        thickness_cut=thickness_cut,
        nb_row=nb_row,
        nb_col=nb_col,
        visualisation_flag=visualisation_flag,
        save_plot_flag=save_plot_flag,
    )

    return percentage_dataframe, layer_boundaries_dataframe


def densities_from_layers(
    image_dataframe: pd.DataFrame,
    layers: list,
    thickness_cut: float = 50,
    alpha: float = 0.001,
) -> list:
    """
    computes the densities for each layers
    Args:
        image_dataframe: pandas.Datatframe that contains the cells
                features the RF_prediction feature set
        thickness_cut: float: The thikness of the cut in um unit
        alpha: flaot:value to influence the gooeyness of the border. Smaller
                  numbers don't fall inward as much as larger numbers. Too large,
                  and you lose everything!
    """
    nb_cell_per_slide = []
    polygons = []
    cells_pos_list = []

    for layer in layers:
        df_layer = image_dataframe[image_dataframe.RF_prediction == layer]
        df_layer = df_layer[df_layer.exclude_for_density == False]

        cells_pos = df_layer[["Centroid X µm", "Centroid Y µm"]].to_numpy()
        points = cells_pos

        if layer == "Layer 1":
            concave_hull = alphashape.alphashape(points, alpha=alpha / 10)
        else:
            # try:
            concave_hull = alphashape.alphashape(points, alpha=alpha)
            # except TypeError:
            #    concave_hull = alphashape.alphashape(points, alpha=0)

        if isinstance(concave_hull, MultiPolygon):
            concave_hull = get_bigger_polygon(concave_hull)
        polygons.append(concave_hull)
        cells_pos_list.append(cells_pos)

        inside_points = get_inside_points(concave_hull, cells_pos)
        nb_cell_per_slide.append(inside_points.shape[0])

    densities = compute_cell_density_per_layer(
        nb_cell_per_slide, polygons, thickness_cut / 1e3
    )

    return densities, cells_pos_list, polygons


def compute_cell_density(nb_cell_per_slide, split_polygons, z_length):
    """
    Computes density as function of brain percentage of depth
    Args:
        nb_cell_per_slide: list of int
        split_polygons:list of shapely polygons representing S1 layers as function if brain depth
        z_length: float ( thickness of the cut over z axis (mm)
    Returns:
         tuple:
        -  depth_percentage: list of float representing the percentage of brain depth
        -  densities: list of float representing the number of cell by mm3
    """
    nb_cells = []
    densities = []

    for nb_cell, polygon in zip(nb_cell_per_slide, split_polygons):
        nb_cells.append(nb_cell)
        densities.append(nb_cell / ((polygon.area / 1e6) * z_length))

    depth_percentage = [i / len(split_polygons) for i in range(len(split_polygons))]

    return depth_percentage, densities, nb_cells


def compute_cell_density_per_layer(nb_cell_per_slide, split_polygons, z_length):
    """
    Computes density as function of brain percentage of depth
    Args:
        nb_cell_per_slide: list of int
        split_polygons:list of shapely polygons representing S1 layers as function if brain depth
        z_length: float ( thickness of the cut over z axis (mm)
    Returns:
         tuple:
        -  depth_percentage: list of float representing the percentage of brain depth
        -  densities: list of float representing the number of cell by mm3
    """
    densities = []
    for nb_cell, polygon in zip(nb_cell_per_slide, split_polygons):
        densities.append(nb_cell / ((polygon.area / 1e6) * z_length))

    return densities


def compute_animal_densities(
    cell_feature_path,
    s1hl_path,
    metadata_path,
    layer_thickness,
    cell_position_file_prefix,
    S1HL_file_sufix,
    db_image_to_exlude_list,
):
    """
    Compute the SH!L density animal by animal
    Args:
        cell_feature_path: (pathlib.Path)
        s1hl_path : (pathlib.Path)
        metadata_path: (pathlib.Path)
        layer_thickness (float)
        cell_position_file_prefix (str)
        S1HL_file_sufix (str)
        db_image_to_exlude_list: (pathlib.Path)

    :return:
    A dictionary : keys animal, values cell density
    """
    densites = defaultdict(list)

    cell_feature_list = glob.glob(
        str(cell_feature_path / cell_position_file_prefix) + "*.csv"
    )
    animal_by_image = get_animal_by_image_id(metadata_path)

    index = 0
    for feature_path in cell_feature_list:
        image_id = get_image_id(feature_path)
        animal = animal_by_image[image_id]
        if image_id in db_image_to_exlude_list:
            continue
        cur_s1hl_path = s1hl_path / (image_id + S1HL_file_sufix + ".csv")

        df_feat = pd.read_csv(feature_path)
        nb_cells = len(df_feat[df_feat.exclude_for_density == False])

        df_s1hl = pd.read_csv(cur_s1hl_path, index_col=0)
        s1hl_points = df_s1hl[["Centroid X µm", "Centroid Y µm"]].to_numpy()
        poly = shapely.Polygon(s1hl_points)
        volume = poly.area * layer_thickness / 1e9

        density = nb_cells / volume
        densites[animal].append(density)
        index += 1

    print(f"INFO: Done {index} images computed")

    return densites
