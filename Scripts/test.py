
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

dataroot = r"C:/Users/Ruben/OneDrive/Bureaublad/data/sets/nuscenes"
#dataroot = r"C:/Users/marni/OneDrive/Documents/BEP 2024/data/sets/nuscenes"
#dataroot = r'C:/Users/Chris/Python scripts/BEP VALDERS/data/sets/nuscenes'

<<<<<<< HEAD
map_name = 'singapore-queenstown'  #'singapore-onenorth'
map_short = 'singapore'
datafile_name = 'reinitialized_data.pkl'
=======
map_name = 'boston-seaport'  #'singapore-onenorth'
map_short =  'singapore' #'Boston'
datafile_name = 'data'
>>>>>>> a69f297b494365c3fca7713b012cd7d4a5cc43f5

map_width = 2979.5
map_height = 2118.1

amount_cones = 8
max_power = 64 # watt
procent = 0.5
LIDAR_RANGE = 100 # 100 meter
OCC_ACCUM = 1 / 8 # full accumulation in 8 samples = 4 sec 
LIDAR_DECAY = 1 # amount of occurrence that goes down per lidar point
probability_threshold = 0.6 

risk_weights = (1, 4, 2) # (0.5, 2, 10) # static, detection, tracking

<<<<<<< HEAD
scene_id = 5
RESOLUTION = 5 # meter
=======
scene_id = 6
RESOLUTION = 0.5 # meter
>>>>>>> a69f297b494365c3fca7713b012cd7d4a5cc43f5

run_detect = True
run_obj = True
run_power = True

plot_layers = True
plot_pointcloud = True
show_pointcloud = False
plot_occ_hist = True
plot_occ = True
plot_risk = True
plot_intermediate_risk = True
plot_power_profile = True


def main(map_short, id, LIDAR_RANGE, RESOLUTION, OCC_ACCUM, LIDAR_DECAY):
    # Entry point for the main simulation function
    print("Starting main function...")

    # Initialize the map object with given parameters
    map_const = Map(dataroot, map_name, map_width, map_height, id, LIDAR_RANGE, RESOLUTION, OCC_ACCUM, LIDAR_DECAY)
    map_var = Map(dataroot, map_name, map_width, map_height, id, LIDAR_RANGE, RESOLUTION, OCC_ACCUM, LIDAR_DECAY)
    maps = [map_const, map_var]

    # Create a folder structure to save the run results and plots
    scene_name = os.path.join("Runs", map_short, f"scene {id} res={RESOLUTION}")
    run_folder_cons = os.path.join(scene_name, 'Constant Power')
    run_folder_var = os.path.join(scene_name, 'Variable Power')
    run_folders = [run_folder_cons, run_folder_var]

    scene_data_paths = []

    for run, run_folder in enumerate(run_folders):
        os.makedirs(run_folder, exist_ok=True)  # Ensure the directory exists
        # Paths for data and specific plots
        scene_data_paths.append(os.path.join(run_folder, datafile_name))
        os.makedirs(scene_data_paths[run])
    risk = Risk(risk_weights)

    # assigns layers for both simulation, the map var that calls it does not matter
    Map.assign_layers(scene_data_paths, maps, prnt=False)
    Visualise.plot_layers(maps[0].grid, run_folder_cons)




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