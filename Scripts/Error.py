
class Error:
    def __init__(self, maps):
        self.maps = maps

    def found_objects_difference(self):
        """
        Compute the average difference in total objects between maps[0] and maps[1] over time.
        """
        avg_obj_map0 = sum(self.maps[0].grid.total_obj) / len(self.maps[0].grid.total_obj)
        avg_obj_map1 = sum(self.maps[1].grid.total_obj) / len(self.maps[1].grid.total_obj)
        return avg_obj_map0 - avg_obj_map1

    def object_sev_difference(self):
        """
        Compute the average difference in total object severity between maps[0] and maps[1] over time.
        """
        avg_sev_map0 = sum(self.maps[0].grid.total_obj_sev) / len(self.maps[0].grid.total_obj_sev)
        avg_sev_map1 = sum(self.maps[1].grid.total_obj_sev) / len(self.maps[1].grid.total_obj_sev)
        return avg_sev_map0 - avg_sev_map1

    def total_avg_occ_difference(self):
        """
        Compute the average occupancy difference across all timesteps using grid.avg_occ.
        """
        avg_occ_map0 = self.maps[0].grid.avg_occ
        avg_occ_map1 = self.maps[1].grid.avg_occ
        total_diff = sum(o0 - o1 for o0, o1 in zip(avg_occ_map0, avg_occ_map1))
        return total_diff / len(avg_occ_map0) if len(avg_occ_map0) > 0 else 0
    
    def avg_occ_difference_in_range(self, range_min, range_max):
        """
        Compute the average occupancy difference for cells within a specified range.
        The ego position is updated for each timestep.
        """
        total_diff = 0
        count = 0

        for t in range(len(self.maps[0].ego_positions)):  # Iterate over each timestep
            ego = self.maps[0].ego_positions[t]  # Update ego position for the current timestep

            # Get the cells within the current and smaller range
            smaller_range_cells = set(self.maps[0].grid.circle_of_interrest(range_min, ego))
            current_range_cells = set(self.maps[0].grid.circle_of_interrest(range_max, ego))
            exclusive_cells_in_range = current_range_cells - smaller_range_cells

            # Filter out empty cells
            exclusive_cells_in_range = [cell for cell in exclusive_cells_in_range if cell.layer != 'empty']

            # Compute differences in occupancy values for the current timestep
            for cell0, cell1 in zip(exclusive_cells_in_range, exclusive_cells_in_range):
                total_diff += cell0.occ[t] - cell1.occ[t]
                count += 1

        return total_diff / count if count > 0 else 0

    def avg_occ_difference_0_20m(self):
        return self.avg_occ_difference_in_range(0, 20)

    def avg_occ_difference_20_40m(self):
        return self.avg_occ_difference_in_range(20, 40)

    def avg_occ_difference_40_60m(self):
        return self.avg_occ_difference_in_range(40, 60)

    def avg_occ_difference_60_80m(self):
        return self.avg_occ_difference_in_range(60, 80)

    def avg_occ_difference_80_100m(self):
        return self.avg_occ_difference_in_range(80, 100)
    
    def save_results_to_file(self, file_path):
        """
        Calls all error computation functions and saves their results to a text file at the specified location.
        """
        results = [
            f"Found objects difference: {self.found_objects_difference()}",
            f"Object severity difference: {self.object_sev_difference()}",
            f"Total average Occupancy Uncertainty difference: {self.total_avg_occ_difference()}",
            f"Average Occupancy Uncertainty difference 0-20m: {self.avg_occ_difference_0_20m()}",
            f"Average Occupancy Uncertainty difference 20-40m: {self.avg_occ_difference_20_40m()}",
            f"Average Occupancy Uncertainty difference 40-60m: {self.avg_occ_difference_40_60m()}",
            f"Average Occupancy Uncertainty difference 60-80m: {self.avg_occ_difference_60_80m()}",
            f"Average Occupancy Uncertainty difference 80-100m: {self.avg_occ_difference_80_100m()}"
        ]

        with open(file_path, "w") as file:
            file.write("\n".join(results))
        
        print(f"Error results saved as {file_path}")

    def load_results_from_file(self, file_path):
        """
        Reads the results from a text file and returns them as a dictionary.
        """
        results = {}
        with open(file_path, "r") as file:
            for line in file:
                key, value = line.strip().split(": ", 1)
                results[key] = float(value)
        return results