
# This class handles the risk function(s)
class Risk:

    _instance = None  # Class-level attribute to store the singleton instance

    weights = (1, 1, 1)

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            # Create the instance if it doesn't exist
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Initialize only if this is the first time the instance is created
        if not hasattr(self, "initialized"):  # Prevent re-initialization
            self.initialized = True


    

    def CalcRisk(self, map, weights, i):
        """
        Calculate the total risk as a weighted sum of static_risk, detect_risk, and track_risk.

        :param weights: Tuple of three weights (w_static, w_detect, w_track)
        :return: Total calculated risk
        """
        if len(weights) != 3:
            raise ValueError("Weights must be a tuple of length 3 (w_static, w_detect, w_track).")
        
        w_static, w_detect, w_track = weights
        

        for cells in map.grid.grid:
            for cell in cells:
                cell.total_risk[i] = w_static * cell.static_risk + w_detect * cell.detect_risk[i] + w_track * cell.track_risk[i]
        

    # Is handled in the layer assignment code
    def StaticRisk(self):
        return

    def DetectionRisk(self):
        return

    def TrackingRisk(self):
        return
