"""
visualisation module
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
from math import pi
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

from layer_recognition.geometry import compute_cells_polygon_level
from layer_recognition.utilities import bimodal, gauss

# plt.rcParams["font.family"] = "Arial"
layers_color = {
    "Layer 1": "#ff0000",
    "Layer 2": "#ff0099",
    "Layer 3": "#cc00ff",
    "Layer 2/3": "#751402",
    "Layer 4": "#3300ff",
    "Layer 5": "#0066FF",
    "Layer 6 a": "#00ffff",
    "Layer 6 b": "#00ff66",
}


def get_layer_colors(values):
    """
    Get the specific colors for the layers depending on number of layers (layer 2/3 merged or not)
    Args:
        values:(list of str): Layer names
    Returns:
        list of RGB values (hex)
    """
    colors = list(layers_color.values())

    if len(values) == 6:  # Merged 2/3
        return np.take(colors, [0, 3, 4, 5, 6, 7])

    if len(values) == 7:  # distinguish 2 and 3
        return np.take(colors, [0, 1, 2, 4, 5, 6, 7])

    return None


from PIL import ImageColor


def get_color(distiguish=True, return_type="dict", return_unit="hex"):
    if distiguish:
        layers_color = {
            "Layer 1": "#ff0000",
            "Layer 2": "#ff0099",
            "Layer 3": "#cc00ff",
            "Layer 4": "#3300ff",
            "Layer 5": "#0066FF",
            "Layer 6 a": "#00ffff",
            "Layer 6 b": "#00ff66",
        }
    else:
        layers_color = {
            "Layer 1": "#ff0000",
            "Layer 2/3": "#751402",
            "Layer 4": "#3300ff",
            "Layer 5": "#0066FF",
            "Layer 6 a": "#00ffff",
            "Layer 6 b": "#00ff66",
        }
    layers_color_int = {}
    layers_color_float = {}

    for key, value in layers_color.items():
        layers_color_int[key] = list(np.array(ImageColor.getcolor(value, "RGB")))
        layers_color_float[key] = list(
            np.array(ImageColor.getcolor(value, "RGB")) / 255
        )

    if return_type == "dict":
        if return_unit == "hex":
            return layers_color
        elif return_unit == "float":
            return layers_color_float
        elif return_unit == "int":
            return layers_color_int

    if return_type == "list":
        if return_unit == "hex":
            return list(layers_color.values())
        elif return_unit == "float":
            return list(layers_color_float.values())
        elif return_unit == "int":
            return list(layers_color_int.values())

    raise (ValueError("return_type or return_unit not valid"))


def plot_segment(line, color="black"):
    """
    Plot a segment of line
    Args:
        line:(list of floats) shape (2,2) representing the XY positions od two points
        color: list of str
    """
    pt1 = line[0]
    pt2 = line[1]
    plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], linestyle="-", c=color)


def plot_densities(
    percentages,
    densities,
    output_path=None,
    image_name="",
    visualisation_flag=False,
    save_plot_flag=False,
):
    """
    Plot the density per layers that represent the brain depth
    Args:
        percentages: list of brain depth percentage (float)
        layers_names: list of str
        densities:  list of float (nb cells / mm3)
        image_name:
    """

    # fig, ax = plt.subplots()
    plt.plot(densities, np.array(percentages) * 100)
    plt.gca().set_xlabel("Cell density (cells/mm3)")
    plt.gca().set_ylabel("percentage of depth (%)")
    plt.gca().invert_yaxis()

    title = "Cell densities as function of percentage of depth"
    if image_name:
        title = image_name + " " + title
    plt.title(title)

    if visualisation_flag:
        plt.show()
    elif save_plot_flag:
        file_path = output_path + "/" + image_name + "_density_per_animal_.svg"
        plt.savefig(file_path, dpi=150)
    plt.clf()


def plot_split_polygons_and_cell_depth(
    split_polygons,
    s1_coordinates,
    cells_centroid_x,
    cells_centroid_y,
    excluded_cells_centroid_x=None,
    excluded_cells_centroid_y=None,
    vertical_lines=None,
    horizontal_lines=None,
    output_path=None,
    image_name=None,
    visualisation_flag=False,
    save_plot_flag=False,
):
    """
    Plot splitted S1 polgygons and cell coordiantes and depth
    Args:
        split_polygons: list of shapely polygons representing S1 layers as
        function if brain depth
        s1_coordinates:(np.array) of shape (nb_vertices, 2) containing S1
        polygon coordinates (mm
        cells_centroid_x: np.array of shape (number of cells, ) of type float
        cells_centroid_y: np.array of shape (number of cells, ) of type float
    """
    cells_polygon_level = compute_cells_polygon_level(
        split_polygons, cells_centroid_x, cells_centroid_y
    )
    colors_depth = {}
    for depth in np.unique(cells_polygon_level):
        colors_depth[depth] = np.random.rand(1, 3)

    colors = []
    for depth in cells_polygon_level:
        colors.append(colors_depth[depth])

    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    plt.axis("equal")
    plt.gca().invert_yaxis()
    for polygon in split_polygons:
        plt.plot(*polygon.exterior.xy, c="black")
    plt.plot(s1_coordinates[:, 0], s1_coordinates[:, 1], "r")
    plt.scatter(
        cells_centroid_x,
        cells_centroid_y,
        c="blue",
        s=10,
        alpha=0.5,
        label="preserved cells",
    )
    if excluded_cells_centroid_x is not None and excluded_cells_centroid_y is not None:
        plt.scatter(
            excluded_cells_centroid_x,
            excluded_cells_centroid_y,
            c="red",
            s=1,
            label="exluded cells",
        )
    if vertical_lines:
        for line in vertical_lines[1:-1]:
            line = line.coords
            plot_segment(line)
    if horizontal_lines:
        for line in horizontal_lines:
            x = line.xy[0]
            y = line.xy[1]
            for i in range(0, len(x), 1):
                plt.plot(x[i : i + 2], y[i : i + 2], "-", linewidth=1, c="black")
    title = "Somatosensory cortex. Each layer represents a percentage of depth \
             following the top of the SSX"
    if image_name:
        title = image_name + " " + title
    plt.title(title)
    plt.xlabel("X coordinates (um)")
    plt.ylabel("Y coordinates (um)")
    plt.legend()
    if visualisation_flag:
        plt.show()
    elif save_plot_flag:
        file_path = output_path + "/" + image_name + "_split_polygons_per_animal_.svg"
        print(f"plt.savefig {file_path}")
        plt.savefig(file_path)
    plt.clf()


def plot_raw_data(top_left, top_right, layer_points, image_name=""):
    """
    Display raw data cells and top line:
        - The original cells coordinates with top_line
        - The main cluster per layers cells with top_line
        - The rotated main cluster per layers cells with rotated top_line
    Args:
        top_left:
        top_right:
        layer_points:
    """
    x_values = [top_left[0], top_right[0]]
    y_values = [top_left[1], top_right[1]]

    plt.figure(figsize=(8, 6))
    plt.gca().invert_yaxis()
    plt.plot(x_values, y_values, c="black")
    for xy in layer_points.values():
        plt.scatter(xy[:, 0], xy[:, 1], s=6, alpha=1.0)
    plt.title(image_name + " Raw data cells (one color per layer)")
    plt.show()
    plt.clf()


def plot_cluster_cells(top_left, top_right, layer_clustered_points, image_name=""):
    """
    Display cells from the main cluster for ech layer:
        - The original cells coordinates with top_line
        - The main cluster per layers cells with top_line
        - The rotated main cluster per layers cells with rotated top_line
    Args:
        top_left:
        top_right:
        layer_points:
        layer_clustered_points:
        rotated_top_line:
        layer_rotatated_points:
    """
    x_values = [top_left[0], top_right[0]]
    y_values = [top_left[1], top_right[1]]

    plt.figure(figsize=(8, 6))

    # CELLS from the main cluster (after DBSCAN)
    plt.gca().invert_yaxis()
    for xy in layer_clustered_points.values():
        plt.scatter(xy[:, 0], xy[:, 1], s=6, alpha=1.0)
        plt.title(image_name + " After removing points outside the clusters")
        plt.plot(x_values, y_values, c="black")
    plt.show()
    plt.clf()


def plot_rotated_cells(rotated_top_line, layer_rotatated_points, image_name=""):
    """
    Display rotated cell and top line:
        - The original cells coordinates with top_line
        - The main cluster per layers cells with top_line
        - The rotated main cluster per layers cells with rotated top_line
    Args:
        rotated_top_line:
        layer_rotatated_points:
    """

    plt.figure(figsize=(8, 6))
    # ROTATED CELLS fron the main cluster
    plt.gca().invert_yaxis()
    x_values = [rotated_top_line[0][0], rotated_top_line[1][0]]
    y_values = [rotated_top_line[0][1], rotated_top_line[1][1]]
    plt.plot(x_values, y_values, c="black")
    for xy in layer_rotatated_points.values():
        if xy.shape[0] > 0:
            plt.scatter(xy[:, 0], xy[:, 1], s=6, alpha=1.0)
    plt.title(image_name + " After clustering and rotation")
    plt.show()
    plt.clf()


def plot_layers_bounderies(
    cells_rotated_df,
    boundaries_bottom,
    y_origin,
    image_name,
    output_path,
    visualisation_flag=False,
):
    """
    Display layers boundaries
    Args:
        cells_rotated_df:
        boundaries_bottom: list [0] -> absolute boundary [1] -> percentage boundary
        y_origin:
        image_name:
        output_path:
        visualisation_flag:
    """
    layers_name = set(cells_rotated_df.Class)
    plt.figure(figsize=[8, 8])
    plt.gca().invert_yaxis()
    xmin = cells_rotated_df["Centroid X µm"].to_numpy(dtype=float).min()
    xmax = cells_rotated_df["Centroid X µm"].to_numpy(dtype=float).max()
    half_letter_size = 10
    xmean = (xmax + xmin) / 2
    for layer in layers_name:
        df_layer = cells_rotated_df[cells_rotated_df.Class == layer]
        sc = plt.scatter(
            df_layer["Centroid X µm"].to_numpy(dtype=float),
            df_layer["Centroid Y µm"].to_numpy(dtype=float) - y_origin,
            s=2,
        )
        border_cells = df_layer[df_layer["border"] == True]
        col = sc.get_facecolors()[0].tolist()
        plt.scatter(
            border_cells["Centroid X µm"].to_numpy(dtype=float),
            border_cells["Centroid Y µm"].to_numpy(dtype=float) - y_origin,
            s=10,
            color=col,
        )
        y = boundaries_bottom[layer][0] + half_letter_size
        plt.hlines(y, xmin, xmax, color="red")
        plt.text(xmean, y - 4 * half_letter_size, layer, size="xx-large")
    plt.hlines(0, xmin, xmax, color="black")
    plt.title(image_name + " Layers bottom boundaries (um)")
    plt.xlabel("X cells' coordinates (um)")
    plt.ylabel("Cells distance from Layer1 top coordinate (um)")
    if visualisation_flag:
        plt.show()
    else:
        file_path = output_path + "/" + image_name + ".svg"
        plt.savefig(file_path, dpi=150)
    plt.clf()


def plot_layer_per_animal(
    dataframe, layers_name, image_name, output_path, visualisation_flag
):
    """
    Plot the layers boundary for an animal
    Args:
        dataframe:(pd.Dataframe)) dataframe that contains layers bottom position by layer
        layers_name:(list(str)) list of layer names
        image_name:(str)
        output_path:(str)
        visualisation_flag(bool) If set display the figure otherwise save it.
    """
    plt.figure(figsize=(8, 8))
    for layer_name in layers_name:
        layer_df = (
            dataframe[dataframe["Layer"] == layer_name]
            .groupby(["animal", "Layer"], as_index=False)[
                "Layer bottom (um). Origin is top of layer 1"
            ]
            .mean()
        )
        positions = layer_df["Layer bottom (um). Origin is top of layer 1"]
        layer_animal = layer_df["animal"]
        plt.scatter(layer_animal, positions, label=layer_name, s=100)
        plt.xlabel("Animal id")
        plt.xticks(rotation=90)
        plt.ylabel("Layer bottom. Origin is top of " + layer_name + " (um)")
        plt.gca().legend()
    plt.gca().set_title("Layer boundaries by input animal")
    if visualisation_flag:
        plt.show()
    else:
        file_path = output_path + "/" + image_name + "layer_per_animal_.svg"
        plt.savefig(file_path, dpi=150)
    plt.clf()


def plot_densities_by_layer(
    layers, layers_densities, image_name, output_path, visualisation_flag=False
):
    """
    Make a horizontal bar plot that represent the cell densities by layer
    Args:
        layers:(list(str)) list of layers name
        layers_densities:(list(float)) of layers cell density
        image_name:(str)
        output_path:(str)
        visualisation_flag(bool) If set display the figure otherwise save it.
    """
    y_pos = np.arange(len(layers))
    plt.barh(y_pos, layers_densities, align="center")
    plt.gca().set_yticks(y_pos, labels=layers)
    plt.title(image_name + " Cells density by layers (nb cell / mm3)")
    plt.gca().invert_yaxis()
    if visualisation_flag:
        plt.show()
    else:
        file_path = output_path + "/" + image_name + "_densities_by_layer.svg"
        plt.savefig(file_path, dpi=150)
    plt.clf()


def plot_layers(
    cells_pos_list, polygons, image_name, alpha, output_path, visualisation_flag=False
):
    """
    Plots the layers represented by polygons and the cells that they contain.
    Args:
        cells_pos_list:(list(np.array)) of shape(N, 2) containing cells
        polygons:(list(Polygon)) that represents the layers
        image_name:(str)
        alpha:(float) The alpha parameter used to build the alphashape
        output_path:(str)
        visualisation_flag(bool) If set display the figure otherwise save it.
    """
    colors = get_layer_colors(polygons)
    for cells_pos, polygon, color in zip(cells_pos_list, polygons, colors):
        plt.scatter(cells_pos[:, 0], cells_pos[:, 1], s=1, color=color)
        try:
            x, y = polygon.exterior.xy
            plt.plot(x, y, color=color)
        except AttributeError:
            pass
    plt.title(f"{image_name} Layer polygon for alpha={alpha}")
    plt.gca().invert_yaxis()
    plt.gca().axis("equal")
    if visualisation_flag:
        plt.show()
    else:
        file_path = output_path + "/" + image_name + "_layer_from_points.svg"
        plt.savefig(file_path, dpi=150)
    plt.clf()


class AnchoredHScaleBar(matplotlib.offsetbox.AnchoredOffsetbox):
    """size: length of bar in data units
    extent : height of bar ends in axes unit
    """

    def __init__(
        self,
        size=1,
        extent=0.03,
        label="",
        loc=2,
        ax=None,
        pad=0.4,
        borderpad=0.5,
        ppad=0,
        sep=2,
        prop=None,
        frameon=True,
        linekw={},
        **kwargs,
    ):
        if not ax:
            ax = plt.gca()
        trans = ax.get_yaxis_transform()
        size_bar = matplotlib.offsetbox.AuxTransformBox(trans)
        line = Line2D([0, 0], [size, 0], **linekw)
        hline1 = Line2D([-extent / 2.0, extent / 2.0], [0, 0], **linekw)
        hline2 = Line2D([-extent / 2.0, extent / 2.0], [size, size], **linekw)
        size_bar.add_artist(line)
        size_bar.add_artist(hline1)
        size_bar.add_artist(hline2)

        txt = matplotlib.offsetbox.TextArea(label, textprops={"size": 12})
        self.vpac = matplotlib.offsetbox.VPacker(
            children=[size_bar, txt], align="center", pad=ppad, sep=sep
        )
        matplotlib.offsetbox.AnchoredOffsetbox.__init__(
            self,
            loc,
            pad=pad,
            borderpad=borderpad,
            child=self.vpac,
            prop=prop,
            frameon=frameon,
            **kwargs,
        )


def plots_cells_size_per_layers(area_dataframe, output_path=None):
    """
    Make a histogram that represent the cell mean diameters (um) by layer
    Args:
        area_dataframe:(pands.Dataframe)
        output_path:(str)
        visualisation_flag:(bool) If True display histogram otherwise save to output_path
    """

    labels = ["L1 ", "L2 ", "L3 ", "L4 ", "L5 ", "L6a ", "L6b "]
    ratios = [0.6, 1.2, 2.6, 1.5, 2.7, 3.1, 0.7]
    _, axes = plt.subplots(7, figsize=(5, 20), sharex=True, gridspec_kw={"height_ratios": ratios})

    
    layers = np.unique(area_dataframe.RF_prediction)
    for i, layer in enumerate(layers):
        layer_area_dataframe = area_dataframe[area_dataframe.RF_prediction == layer]
        areas = layer_area_dataframe["Area µm^2"].to_numpy()
        diameters = np.sqrt((areas / pi)) * 2
   
        y,x,_=axes[i].hist(diameters, 100, alpha=.4, color=layers_color[layer])
        x=(x[1:]+x[:-1])/2 # for len(x)==len(y)
        expected = (diameters.mean()-diameters.std(), 0.2, 100,
                    diameters.mean()+diameters.std(), 0.2, 100)
        params, cov = curve_fit(bimodal, x, y, expected)
        sigma=np.sqrt(np.diag(cov))
        x_fit = np.linspace(x.min(), x.max(), 500)

        y_pop_1 = gauss(x_fit, *params[:3])
        axes[i].plot(x_fit, y_pop_1, color='black', lw=.4, ls="-", label='cell pop 1')
        axes[i].vlines(params[0], 0, y_pop_1.max(), colors='red',  lw=1, ls=":")
        y_pop_2 = gauss(x_fit, *params[3:])
        axes[i].plot(x_fit, y_pop_2, color='black', lw=.4, ls="-", label='cell pop 2')
        axes[i].vlines(params[3], 0, y_pop_2.max(), colors='red', lw=1, ls=":")

        

        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)
        axes[i].spines["bottom"].set_visible(False)
        axes[i].spines["left"].set_visible(False)
        #axes[i].set_ylim(bottom=0, top=1e4 * ratios[i])
        axes[i].set_yticklabels([])
        axes[i].set_yticks([])
        axes[i].set_ylabel(labels[i], rotation=0, fontsize=12)
        axes[i].tick_params(axis="x", labelsize=12)


    '''
    for i in range(6):
        axes[i].set_xticks([])
    '''
    scalebar = AnchoredHScaleBar(
        size=5000,
        label="5000",
        loc="lower right",
        frameon=False,
        pad=0,
        sep=5,
        linekw={"color": "black"},
    )

    axes[5].add_artist(scalebar)

    axes[0].set_title("Cells mean diameter (µm)", fontsize=12)
    print(output_path)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)


 

def plots_cells_size(
    area_dataframe, output_path=None, save_plot_flag=False, visualisation_flag=False
):
    """
    Make a histogram that represent the cell mean diameters (um)
    Args:
        area_dataframe:(pands.Dataframe)
        output_path:(str)
    """
    areas = area_dataframe["Area µm^2"]
    diameters = np.sqrt((areas / pi)) * 2
    _ = plt.hist(diameters, bins=100)
    plt.ylabel("Cell count")
    plt.xlabel("Cell area (µm^2)")
    plt.title("Cells area (µm^2)")

    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels([f"{x:.1e}" for x in current_values])
    if visualisation_flag:
        plt.show()
    elif output_path is not None and save_plot_flag:
        plt.savefig(output_path, dpi=150)


def plots_layer_thickness(
    thickness_dataframe, output_path=None, visualisation_flag=False
):
    """
    Make a bar that represent the layers thinkness mean and std value (um)
    Args:
        thikness_dataframe:(pands.Dataframe)
        output_path:(str)
        visualisation_flag(bool) If set display the figure otherwise save it.
    """

    thickness_mean = thickness_dataframe.thickness_mean.to_list()

    all_animal_thickness_mean = []
    all_animal_thickness_std = []
    for values in thickness_mean:
        all_animal_thickness_mean.append(np.mean(values))
        all_animal_thickness_std.append(np.std(values))

    layers_label = [
        "Layer 1",
        "Layer 2",
        "Layer 3",
        "Layer 4",
        "Layer 5",
        "Layer 6 a",
        "Layer 6 b",
    ]
    color_bar = get_layer_colors(layers_label)

    # Création des barres d'erreurs
    fig, ax = plt.subplots(figsize=(10, 7))
    print(f"INFO: Layers thickness mean and std:")
    for layer, mean, std in zip(
        layers_label, all_animal_thickness_mean, all_animal_thickness_std
    ):
        print(f" {layer}: {mean} {std}")
    plt.barh(
        layers_label,
        all_animal_thickness_mean,
        xerr=all_animal_thickness_std,
        capsize=2,
        color=color_bar,
    )

    plt.ylabel("layer thickness (µm)")

    index = 0
    label = "animal mean"
    for values in thickness_mean:
        for thickness in values:
            if label is not None:
                plt.scatter(
                    thickness,
                    index + (np.random.rand() / 2) - 0.25,
                    s=5,
                    c="orange",
                    label=label,
                )
                label = None
            else:
                plt.scatter(
                    thickness, index + (np.random.rand() / 2) - 0.25, s=5, c="orange"
                )
        index += 1

    plt.gca().invert_yaxis()
    if visualisation_flag:
        plt.show()
    else:
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0)


def plot_cell_density_by_animal(
    densites, layers, output_path, visualisation_flag=False
):
    """
    Make a bars that represent the S1HL brain region cell density (cell/mm3) by animal
    Args:
        densites:(dictionary)
        layers:(list)
        output_path:(str)
        visualisation_flag:(bool) If True display histogram otherwise save to output_path
    """
    densities_dict_mean = defaultdict(list)
    densities_dict_std = defaultdict(list)
    # fig, ax = plt.subplots(figsize =(10, 7))

    for animal, layers_densities in densites.items():
        for layer_name, layer_densities in layers_densities.items():
            densities_dict_mean[layer_name].append(np.mean(layer_densities))
            densities_dict_std[layer_name].append(np.std(layer_densities))
    densities_mean = list()
    densities_std = list()

    for mean_values, std_valeus in zip(
        densities_dict_mean.values(), densities_dict_std.values()
    ):
        densities_mean.append(np.mean(mean_values))
        densities_std.append(np.std(std_valeus))
    bar_colors = get_color(distiguish=True, return_type="list", return_unit="float")
    fig = plt.figure(figsize=(10, 10))

    plt.barh(layers, densities_mean, xerr=densities_std, capsize=2, color=bar_colors)
    plt.ylabel("cell density (cell/mm3)")
    plt.xlabel("Layers cell density (cell/mm3)")

    current_values = plt.gca().get_xticks()
    _ = plt.gca().set_xticklabels(["{:.1e}".format(x) for x in current_values])

    index = 0
    label = "animal mean"
    for key, values in densities_dict_mean.items():
        for density in values:
            if np.isnan(density) == False:
                if label is not None:
                    plt.scatter(
                        density,
                        index + (np.random.rand() / 2) - 0.25,
                        s=5,
                        c="orange",
                        label=label,
                    )
                    label = None
                else:
                    plt.scatter(
                        density, index + (np.random.rand() / 2) - 0.25, s=5, c="orange"
                    )
            else:
                print(f"INFO {key} density is NAN")
        index += 1

    plt.gca().invert_yaxis()
    plt.legend()

    if visualisation_flag:
        plt.show()
    else:
       fig.savefig(output_path, bbox_inches="tight", pad_inches=0)


    """
    print(density_animal_dataframe)
    fig, ax = plt.subplots(figsize=(10, 7))
    animal = density_animal_dataframe.animal.astype(str).to_list()
    densities_mean = density_animal_dataframe.densities_mean.to_list()
    densities_std = density_animal_dataframe.densities_std.to_list()
    plt.bar(animal, densities_mean, yerr=densities_std, capsize=2)
    plt.title("S1HL cell density (cell/mm3)")
    plt.xlabel("Layers")
    plt.ylabel("cell density (cell/mm3)")
    if visualisation_flag:
        plt.show()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    """
