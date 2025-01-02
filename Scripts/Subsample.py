from nuscenes.nuscenes import NuScenes 
import numpy as np
import os
import math

erx = 0.9 # receiver optics effeciency
etx = 0.9 # emmitter optics effeciency
n = 1 #target reflectivity
D = 25*pow(10,-3) #diameter lens 25 mm
Aovx = 1/np.pi #1 graden in radialen
Aovy = 1/np.pi #1 graden in radialen
phi_amb = 13.27 #W/m^2 gekozen via tabel want test wast delta labda = 50 nm
Nshots = 1
Ro = 0.9 #We kiezen een ADP lidar met een golflengte van 1550 nm
M = 30
F = 7
Bn = 1* 10**6 #Bandwidth 1 MHz
Id = 150*10**-9 # Dark current 150 nA
Rf = 10*10**3 #feed resistance 10 K ohm
vamp = 28*10**-9 # amplifier input voltage noice density 28 nV/sqrt(Hz)
T =293 #20 graden celsius in K
Kb = 1.380649*10**-23 # boltzmann constant
e = 1.602*10**-19 # elementaire lading
P_false = 10**-4 # false trigger mochten klein zetten
car_height = 1.8

class subsample():
    def __init__(self, map, n_cones):
        self._power = None
        self._sample = None
        self.oud = None
        self.file = None
        self.dataroot = map.dataroot
        self.map = map
        self.nusc = map.nusc
        self.lidar_punt= None
        self.n_cones = n_cones
        self.max_range = map.range
        self._scene_id = None
        self.count = None
        self.count_new = None

      
    def update(self, sample, sample_index, scene_id, power):
        self._sample = sample
        self._sampleindex = sample_index
        self._scene_id = scene_id
        self._power = power
        if self._sample != self.oud: # alleen runnen als sample veranderd
            self.lidarpoint = []
            self.subsamp = []
            self.count = 0
            self.count_new = 0
            info = self.nusc.get('sample', self._sample)
            info = self.nusc.get('sample_data', info['data']['LIDAR_TOP'])
            
            sen_info = self.nusc.get('calibrated_sensor', info['calibrated_sensor_token'])
            
            info_2 = self.nusc.get('ego_pose', info['ego_pose_token'])
            
            rot_2 = np.arctan2((2*(sen_info['rotation'][0]*sen_info['rotation'][3]+sen_info['rotation'][1]*sen_info['rotation'][2])),(1-2*(sen_info['rotation'][3]**2+sen_info['rotation'][2]**2)))
            rot = np.arctan2((2*(info_2['rotation'][0]*info_2['rotation'][3]+info_2['rotation'][1]*info_2['rotation'][2])),(1-2*(info_2['rotation'][3]**2+info_2['rotation'][2]**2))) 
            xy_lidar = np.array([sen_info['translation'][0], sen_info['translation'][1]]).reshape(-1, 1) 
            rot_matrix = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
            rot_matrix_2 = np.array([[np.cos(rot_2), -np.sin(rot_2)], [np.sin(rot_2), np.cos(rot_2)]])

            self.file_get()
            self.lidar_coor(rot_matrix,rot_matrix_2, xy_lidar)
            cones = self.cones()
            #print (self.lidar_punt)
            #print (cones)
            for i in range(self.lidar_punt):
                a,d = self.a_d(self.lidarpoint[i][0],self.lidarpoint[i][1], self.lidarpoint[i][2])
                if a < 0:
                    a = a+360
                if d <= (self.max_range+200):
                    for j, (start_angle, end_angle) in enumerate(cones):
                        if start_angle <= a < end_angle:
                            pro = self.calc_proba(self._power[j], d)
                            if pro >= 0.9:
                                self.subsamp.append((self.lidarpoint[i]))
                                self.count_new +=1
                else: self.count+=1



    def file_get(self): #Deze functie zoekt het bestand van de lidar pointcloud die bij de sample hoort. Vervolgens wordt het volledige pad er naar toe gemaakt.
        info =  self.nusc.get('sample', self._sample)
        info_2 = self.nusc.get('sample_data',info['data']['LIDAR_TOP'])
        self.file = os.path.join(self.dataroot, info_2['filename'])

    def lidar_coor(self, rot_1, rot_2, xy_l):

        som = 0
        
        with open(self.file, "rb") as f:
            number = f.read(4)
            self.lidar_punt = 0
            while number != b"":
                quo, rem = divmod(som,5) #omdat je alleen x en y wilt gebruiken en niet de andere dingen kijk je naar het residu van het item waar die op zit.
                if rem == 0: # als het residu = 0 heb je het x coordinaat en res = 1 is het y-coordinaat
                    x = np.frombuffer(number, dtype=np.float32)[0]
                    number = f.read(4) #leest de volgende bit    
                elif rem ==1:
                    y = np.frombuffer(number, dtype=np.float32)[0]
                    number = f.read(4) #leest de volgende bit
                elif rem ==2:
                    z = np.frombuffer(number, dtype=np.float32)[0]
                    number = f.read(4) #leest de volgende bit
                elif rem == 3:
                    intensity = np.frombuffer(number,dtype=np.float32)[0]
                    number = f.read(4) #leest de volgende bit
                elif rem == 4:
                    xy = np.array([x, y]).reshape(-1, 1)
                    xy_rotated = np.dot(rot_2, xy)
                    xy_rot_2 = xy_rotated+xy_l
                    xy_rot = np.dot(rot_1, xy_rot_2)
                    
                    ring_index = np.frombuffer(number,dtype=np.float32)[0]
                    self.lidarpoint.append((xy_rot[0],xy_rot[1],z,intensity,ring_index))
                    self.lidar_punt += 1
                    number = f.read(4)
                else:
                    number = f.read(4) #leest de volgende bit
                som +=1 # som houdt bij hoeveel items gelezen zijn.

    def a_d (self, x, y, z):
        angle = math.degrees(math.atan2((y),(x)))
        distance = math.sqrt((y)**2 + (x)**2 + (z-car_height)**2)
        return angle, distance
    
    def cones (self):
        angle_step = 360 / self.n_cones
        quadrants = [(i * angle_step, (i + 1) * angle_step) for i in range(self.n_cones)]
        return quadrants
    
    def calc_proba(self, power,r):
        amp_area = np.pi*(D/2)**2
        P_s = (1/(2*np.pi*r**2))*power*erx*etx*n*amp_area
        Aov = 4*r**2*np.tan(Aovx/2)*np.tan(Aovy/2)
        P_b = (1/(2*np.pi*r**2))*phi_amb*amp_area*Aov*erx*n
        SNR = np.sqrt(Nshots*Ro**2*P_s**2)/np.sqrt(2*e*Bn*F*(Ro*(P_s+P_b)+Id)+(Bn/M**2)*(4*Kb*T/Rf + (vamp/Rf)**2))
        Prob = 0.5*math.erfc(np.sqrt(-math.log(P_false))-np.sqrt(SNR+0.5))
        return Prob
    def create_bin_file(self):
        output_file = f"lidar_data_{self._scene_id}_{self._sample}.bin"
        with open(output_file, "wb") as bin_file:
            for point in self.lidarpoint:
                point_array = np.array(point, dtype=np.float32)
            	# Write the array as binary data to the file
                bin_file.write(point_array.tobytes())
        return output_file
