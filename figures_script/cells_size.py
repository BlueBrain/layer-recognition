#!/usr/bin/env python
# coding: utf-8

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

import sys

import pandas as pd

from layer_recognition.visualisation import (
    plots_cells_size,
    plots_cells_size_per_layers,
)

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print(
            "usage: python cell_size.py cell_area_dataframe_path output_directory_path [per_layer]"
        )
        sys.exit()

    cell_area_dataframe_path = sys.argv[1]
    output_figure_path = sys.argv[2]
    area_dataframe = pd.read_csv(cell_area_dataframe_path)

    if len(sys.argv) == 4 and sys.argv[3] == "per_layer":
        plots_cells_size_per_layers(area_dataframe, output_path=output_figure_path)
    else:
        plots_cells_size(
            area_dataframe,
            output_path=output_figure_path,
            save_plot_flag=True,
            visualisation_flag=False,
        )
