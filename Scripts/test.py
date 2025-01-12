
# This is the file that executes all the code

import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from datetime import timedelta
import numpy as np
import re
import Cell
import os
from Grid import Grid 
from Map import Map
from Visualise import Visualise
from Risk import Risk
from Object import Object
from Dectetion import Detect
from Power import power
from Subsample import subsample
from Object_filter import object_filter
from Error import Error

import warnings
from shapely.errors import ShapelyDeprecationWarning

# Suppress ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap

#dataroot = r"C:/Users/Ruben/OneDrive/Bureaublad/data/sets/nuscenes"
#dataroot = r"C:/Users/marni/OneDrive/Documents/BEP 2024/data/sets/nuscenes"
dataroot = r'C:/Users/Chris/Python scripts/BEP VALDERS/data/sets/nuscenes'

map_name = 'boston-seaport'  #'singapore-onenorth'
map_short = 'Boston'

map_width = 2979.5
map_height = 2118.1

amount_cones = 8
max_power = 64 # watt
procent = 0.75
LIDAR_RANGE = 100 # 100 meter
OCC_ACCUM = 1 / 8 # full accumulation in 8 samples = 4 sec 
LIDAR_DECAY = 1 # amount of occurrence that goes down per lidar point


risk_weights = (1, 4, 2) # (0.5, 2, 10) # static, detection, tracking

scene_id = 1
RESOLUTION = 0.5 # meter

run_detect = True
run_obj = True
run_power = True

plot_layers = False
plot_pointcloud = True
show_pointcloud = False
plot_occ_hist = True
plot_occ = True
plot_risk = True
plot_intermediate_risk = True
plot_power_profile = False
n_cones = 8


p_uniform = [8]*n_cones


def main(map_short, id, LIDAR_RANGE, RESOLUTION, OCC_ACCUM, LIDAR_DECAY):
    # Entry point for the main simulation function
    print("Starting main function...")

    # Initialize the map object with given parameters
    map_const = Map(dataroot, map_name, map_width, map_height, id, LIDAR_RANGE, RESOLUTION, OCC_ACCUM, LIDAR_DECAY)
    map_var = Map(dataroot, map_name, map_width, map_height, id, LIDAR_RANGE, RESOLUTION, OCC_ACCUM, LIDAR_DECAY)
    maps = [map_const, map_var]

    # Create a folder structure to save the run results and plots
    run_folder_cons = os.path.join("Runs", map_short, f"scene {id} res={RESOLUTION}", 'Constant Power')
    run_folder_var = os.path.join("Runs", map_short, f"scene {id} res={RESOLUTION}", 'Variable Power')
    run_folders = [run_folder_cons, run_folder_var]

    # create a folder where all the comparison plots are made
    comparison_folder = os.path.join("Runs", map_short, f"scene {id} res={RESOLUTION}")
    os.makedirs(comparison_folder, exist_ok=True)

    power_profile_folder = os.path.join("Runs", map_short, f"scene {id} res={RESOLUTION}", 'Power Profiles')
    os.makedirs(power_profile_folder, exist_ok=True)

    plots_folders = []
    gif_folders = []
    scene_data_paths = []
    plots_folders = []
    
    layer_plot_paths = []
    pointclouds_folders = []
    pointclouds_overlay_folders = []

    risk_plots_folders = []
    occ_folders = []
    occ_hist_folders = []

    for run, run_folder in enumerate(run_folders):
        os.makedirs(run_folder, exist_ok=True)  # Ensure the directory exists

        # Create subdirectories for storing plots and GIFs
        plots_folders.append(os.path.join(run_folder, 'plots'))
        os.makedirs(plots_folders[run], exist_ok=True)

        gif_folders.append(os.path.join(run_folder, 'GIFs'))
        os.makedirs(gif_folders[run], exist_ok=True)

        # Paths for data and specific plots
        scene_data_paths.append(os.path.join(run_folder, "data"))
        layer_plot_paths.append(os.path.join(plots_folders[run], "layers.png"))

        # Subdirectories for specific plot types
        risk_plots_folders.append(os.path.join(plots_folders[run], "risks"))
        pointclouds_folders.append(os.path.join(plots_folders[run], "pointclouds"))
        pointclouds_overlay_folders.append(os.path.join(plots_folders[run], "pointclouds overlay"))
        occ_folders.append(os.path.join(plots_folders[run], "occurrence"))
        occ_hist_folders.append(os.path.join(plots_folders[run], "occurrence histograms"))

        # Ensure subdirectories exist
        os.makedirs(risk_plots_folders[run], exist_ok=True)
        os.makedirs(pointclouds_folders[run], exist_ok=True)
        os.makedirs(pointclouds_overlay_folders[run], exist_ok=True)
        os.makedirs(occ_folders[run], exist_ok=True)
        os.makedirs(occ_hist_folders[run], exist_ok=True)

    # assigns layers for both simulation, the map var that calls it does not matter
    Map.assign_layers(scene_data_paths, maps, prnt=False)
    Visualise.plot_avg_occ(maps, comparison_folder)
    # Usage example (assuming `maps` is a list of two map objects):
    error = Error(maps)
    error.save_results_to_file(os.path.join(comparison_folder, 'error_results.txt'))
    print('Done')  # End of the main function


# This ensures that the code is only executed when the script is run directly
if __name__ == '__main__':
    print("Running as main module...")  # Debugging line
    start_time = time.time()
    main(map_short=map_short, 
        id=scene_id, 
        LIDAR_RANGE=LIDAR_RANGE, 
        RESOLUTION=RESOLUTION, 
        OCC_ACCUM=OCC_ACCUM, 
        LIDAR_DECAY=LIDAR_DECAY)

    run_time = time.time() - start_time
    print(f'\nRunning took {timedelta(seconds=run_time)}')

