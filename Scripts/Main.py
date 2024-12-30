
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


LIDAR_RANGE = 100 # 100 meter
OCC_ACCUM = 1 / 8 # full accumulation in 8 samples = 4 sec 
LIDAR_DECAY = 1 # amount of occurrence that goes down per lidar point

risk_weights = (1, 4, 2) # (0.5, 2, 10) # static, detection, tracking

scene_id = 1
RESOLUTION = 0.5 # meter
run_detect = False
run_obj = False
plot_layers = False
plot_pointcloud = False
show_pointcloud = False
plot_occ_hist = False
plot_occ = False
plot_risk = False

constant_power = True

def main(map_short, id, LIDAR_RANGE, RESOLUTION, OCC_ACCUM, LIDAR_DECAY, constant_power):
    # Entry point for the main simulation function
    print("Starting main function...")

    # Initialize the map object with given parameters
    map = Map(dataroot, map_name, map_width, map_height, id, LIDAR_RANGE, RESOLUTION, OCC_ACCUM, LIDAR_DECAY)

    # Determine the simulation type based on power mode
    if constant_power:
        sim_type = 'Constant Power'
    else:
        sim_type = 'Variable Power'

    # Create a folder structure to save the run results and plots
    run_folder = os.path.join("Runs", map_short, f"scene {id} res={RESOLUTION}", sim_type)
    os.makedirs(run_folder, exist_ok=True)  # Ensure the directory exists

    # Create subdirectories for storing plots and GIFs
    plots_folder = os.path.join(run_folder, 'plots')
    os.makedirs(plots_folder, exist_ok=True)

    gif_folder = os.path.join(run_folder, 'GIFs')
    os.makedirs(gif_folder, exist_ok=True)

    # Paths for data and specific plots
    scene_data_path = os.path.join(run_folder, "data")
    layer_plot_path = os.path.join(plots_folder, "layers.png")

    # Subdirectories for specific plot types
    risk_plots_folder = os.path.join(plots_folder, "risks")
    pointclouds_folder = os.path.join(plots_folder, "pointclouds")
    pointclouds_overlay_folder = os.path.join(plots_folder, "pointclouds overlay")
    occ_folder = os.path.join(plots_folder, "occurrence")
    occ_hist_folder = os.path.join(plots_folder, "occurrence histograms")

    # Ensure subdirectories exist
    os.makedirs(risk_plots_folder, exist_ok=True)
    os.makedirs(pointclouds_folder, exist_ok=True)
    os.makedirs(pointclouds_overlay_folder, exist_ok=True)
    os.makedirs(occ_folder, exist_ok=True)
    os.makedirs(occ_hist_folder, exist_ok=True)

    # Assign layers to the map grid
    map.assign_layer(scene_data_path, prnt=False)

    # Optionally generate and save a plot of the map layers
    if plot_layers:
        Visualise.plot_layers(map.grid, layer_plot_path)

    # Initialize components for risk calculation, object tracking, and detection
    risk = Risk(risk_weights)
    obj = Object(map)
    dec = Detect(map, constant_power=constant_power)

    # Process each sample in the map
    for i, sample in enumerate(map.samples):
        # Update object data if required
        if run_obj:
            map.grid.total_obj[i], map.grid.total_obj_sev[i] = obj.update(sample=sample, x=0, y=0, sample_index=i, prnt=False)

        # Update detection data if required
        if run_detect:
            dec.update(sample=sample, sample_index=i, prnt=False)

        # Save point cloud plots for each sample
        if plot_pointcloud:
            Visualise.save_pointcloud_scatterplot(map, dec.lidarpoint, i, pointclouds_folder, overlay=False)
            Visualise.save_pointcloud_scatterplot(map, dec.lidarpoint, i, pointclouds_overlay_folder, overlay=True)

        print(f"sample {i} complete\n")

    # Normalize risk data and calculate total risk
    risk.normalise_and_calc_risks(map)

    # Retrieve global maxima for visualization scaling
    maxs = map.get_global_max()

    # Update map grid with risk and object metrics, and generate plots for each sample
    for i, sample in enumerate(map.samples):
        map.update(i=i, weights=risk_weights)  # Update grid with calculated weights

        if plot_risk:
            Visualise.plot_risks_maximised(map.grid, i, maxs, risk_plots_folder)

        if plot_occ:
            Visualise.plot_occ(map.grid, i, occ_folder)

        if plot_occ_hist:
            Visualise.plot_occ_histogram(map, i, occ_hist_folder)

    # Save updated map grid with new risk values
    map.save_grid(scene_data_path)

    # Generate summary plots for the simulation
    Visualise.plot_avg_risks(map.grid, plots_folder)
    Visualise.plot_avg_occ(map.grid.avg_occ, 'Average Occurrence', plots_folder)
    Visualise.plot_total_var(map.grid.total_obj, 'Total Objects', plots_folder)
    Visualise.plot_total_var(map.grid.total_obj_sev, 'Total Object severity', plots_folder)
    Visualise.plot_avg_occ_histogram(map, plots_folder)

    # Create GIFs for visualizing results
    Visualise.create_gif_from_folder(risk_plots_folder, os.path.join(gif_folder, 'risks.gif'))
    Visualise.create_gif_from_folder(pointclouds_folder, os.path.join(gif_folder, 'pointcloud.gif'))
    Visualise.create_gif_from_folder(pointclouds_overlay_folder, os.path.join(gif_folder, 'pointcloud_layers.gif'))
    Visualise.create_gif_from_folder(occ_folder, os.path.join(gif_folder, 'occurrence.gif'))
    Visualise.create_gif_from_folder(occ_hist_folder, os.path.join(gif_folder, 'occurrence_hist.gif'))

    # Check the size of the scene data file and warn if it exceeds 100MB
    data_size = os.path.getsize(scene_data_path)
    if data_size > 100000000:  # File size threshold in bytes
        scene_data_path = scene_data_path.replace('\\', '/')  # Ensure path is in a standard format
        try:
            with open('.gitignore', 'r') as f:
                gitignore_contents = f.read().splitlines()
            if not any(scene_data_path in line for line in gitignore_contents):
                print(f'\nDATA FILE {scene_data_path} \nIS TOO BIG FOR GITHUB: ADD IT TO THE GITIGNORE FILE')
        except:
            pass  # Silently handle exceptions if .gitignore is not accessible

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
        LIDAR_DECAY=LIDAR_DECAY, 
        constant_power=constant_power)

    run_time = time.time() - start_time
    print(f'\nRunning took {timedelta(seconds=run_time)}')
