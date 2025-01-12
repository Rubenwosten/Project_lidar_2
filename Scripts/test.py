from Grid import Grid
from Map import Map
import os

def reinitialize_with_layers(original_grid):
    """
    Reinitialize a Grid object, retaining only the layers information for each cell.
    """
    # Create a new Grid object with the same patch, resolution, and scene length
    new_grid = Grid(
        patch=original_grid.patch,
        resolution=original_grid.res,
        scene_length=original_grid.scene_length,
        RANGE=100  # Reuse or adjust RANGE as needed
    )

    # Iterate through the original grid and copy the `layers` into the new cells
    for x in range(original_grid.width):
        for y in range(original_grid.length):
            original_cell = original_grid.grid[x][y]
            new_grid.grid[x][y].layers = original_cell.layers  # Retain layers
            new_grid.grid[x][y].assign_layer()


    # Copy the `has_assigned_layers` attribute
    new_grid.has_assigned_layers = original_grid.has_assigned_layers

    # Reset grid-level risk and occurrence attributes
    new_grid.avg_total_risk = [0] * original_grid.scene_length
    new_grid.avg_static_risk = [0] * original_grid.scene_length
    new_grid.avg_detection_risk = [0] * original_grid.scene_length
    new_grid.avg_tracking_risk = [0] * original_grid.scene_length
    new_grid.avg_occ = [0] * original_grid.scene_length
    new_grid.total_obj = [0] * original_grid.scene_length
    new_grid.total_obj_sev = [0] * original_grid.scene_length
    new_grid.avg_occ_ranges = [[0] * len(original_grid.ranges) for _ in range(original_grid.scene_length)]
    
    new_grid.create_non_empty_grid()

    return new_grid

dataroot = r'C:/Users/Chris/Python scripts/BEP VALDERS/data/sets/nuscenes'

map_name = 'boston-seaport'  #'singapore-onenorth'
map_short = 'Boston'

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

id = 4
RESOLUTION = 0.5 # meter

# Initialize the map object with given parameters
map_const = Map(dataroot, map_name, map_width, map_height, id, LIDAR_RANGE, RESOLUTION, OCC_ACCUM, LIDAR_DECAY)
map_var = Map(dataroot, map_name, map_width, map_height, id, LIDAR_RANGE, RESOLUTION, OCC_ACCUM, LIDAR_DECAY)
maps = [map_const, map_var]

# Create a folder structure to save the run results and plots
run_folder_cons = os.path.join("Runs", map_short, f"scene {id} res={RESOLUTION}", 'Constant Power')
run_folder_var = os.path.join("Runs", map_short, f"scene {id} res={RESOLUTION}", 'Variable Power')
run_folders = [run_folder_cons, run_folder_var]

original_grid = maps[0].load_grid(os.path.join(run_folder_cons, "data"))

# Reinitialize while retaining only the layers
reinitialized_grid = reinitialize_with_layers(original_grid)

# Assign the new grid back to the map and save it
maps[0].grid = reinitialized_grid
maps[1].grid = reinitialized_grid
maps[0].save_grid(os.path.join(run_folders[0], "reinitialized_data.pkl"))
maps[1].save_grid(os.path.join(run_folders[1], "reinitialized_data.pkl"))


