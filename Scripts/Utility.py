# Utility class
from Grid import Grid
from Map import Map

class Utility:

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
