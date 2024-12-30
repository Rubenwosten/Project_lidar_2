import numpy as np


class object_filter:
    def __init__(self, map):
        self.nusc = map.nusc
        self.ego = map.ego_positions
        self._sample = None
        self._sample_index = None
        self._lidar_new = None
        self.oud = None
        self.object_scanned = None
        self.count = None

    @property
    def sample_up(self):
        return self._sample
    
    def update(self, sample, sample_index, lidar_points, lidar_count):
        self._sample = sample
        self._sample_index = sample_index
        self._lidar_new = lidar_points
        
        if self._sample != self.oud:
            info = self.nusc.get('sample', self._sample)
            object_list = info['anns']
            self.object_scanned = []
            self.count = len(object_list)
            for i in range(len(object_list)):
                ans = object_list[i]
                info = self.nusc.get('sample_annotation', ans)
                rot = np.arctan2((2*(info['rotation'][0]*info['rotation'][3]+info['rotation'][1]*info['rotation'][2])),(1-2*(info['rotation'][3]**2+info['rotation'][2]**2)))
                bounding = self.bounding_box(rot, info['size'],info['translation'])
                if info['size'][0] > info['size'][1]:
                    r = info['size'][0]
                else: r = info['size'][1]
                for i in range(lidar_count):
                    on_box = self.object_dec(bounding, (self._lidar_new[i][0]+self.ego[self._sample_index][0],self._lidar_new[i][1]+self.ego[self._sample_index][1]), r)
                    if on_box == True:
                        self.object_scanned.append(ans)
                        break
            self.oud = sample



    def bounding_box (self, rotation, size, translation):
        corners = np.array([
        [-size[0] / 2, -size[1] / 2],  # Top-left
        [ size[0] / 2, -size[1] / 2],  # Top-right
        [ size[0] / 2,  size[1] / 2],  # Bottom-right
        [-size[0] / 2,  size[1] / 2],  # Bottom-left
        ])
        rotation_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation)],
        [np.sin(rotation),  np.cos(rotation)]
        ])
        rotated_corners = np.dot(corners, rotation_matrix) + np.array([translation[0],translation[1]])
        return rotated_corners

    def object_dec(self, bounding, lidar_coor, r):
        dis = np.zeros(4)
        for i in range (4):
            dis[i] = np.sqrt((bounding[i,0]-lidar_coor[0])**2+(bounding[i,1]-lidar_coor[1])**2)
        if np.min(dis) <= r:
            index = np.argmin(dis)
            closest = bounding[index]
            adjacent_corners = [
                    bounding[(index - 1) % 4],  # Previous corner
                    bounding[(index + 1) % 4]  # Next corner   
                ]
            for i in range(2):
                
                y = self.linear_func(closest, (adjacent_corners[i][0],adjacent_corners[i][1]), lidar_coor[0])
                if y is not None and np.isclose(y, lidar_coor[1], atol=1e-6):
                    return True
            return False
        else: return False

    

    def linear_func(self, p1, p2, x):
        """Calculate y-coordinate for a given x using the line through p1 and p2."""
        
        x1, y1 = p1
        x2, y2 = p2
        if x2 - x1 == 0:  # Vertical line
            return None if x != x1 else y1
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return slope * x + intercept

            


