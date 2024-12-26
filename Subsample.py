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
    def __init__(self, power, dataroot, map, n_cones, lidar_range):
        self.power = power
        self._sample = None
        self.oud = None
        self.file = None
        self.dataroot = dataroot
        self.map = map
        self.nusc = map.nusc
        self.lidar_punt= None
        self.n_cones = n_cones
        self.max_range = lidar_range
        self._scene = None



    @property 
    def sample(self): #getter om sample aftelezen
        return self._sample
    
    @sample.setter
    def sample(self, values): #values is een tuple van sample ego_x en ego_y
        self._sample, self._sampleindex, self._scene = values
        if self._sample != self.oud: # alleen runnen als sample veranderd
            self.lidarpoint = []
            self.subsamp = []
            info = self.nusc.get('sample', self._sample)
            rot = info['rotation']
            rot_matrix = np.array([np.cos(rot),-np.sin(rot)],[np.sin(rot),np.cos(rot)])

            self.file_get()
            self.lidar_coor()
            cones = self.cones()
            for i in range(self.lidar_punt):
                a,d = self.a_d(self.lidarpoint[i][0],self.lidarpoint[i][1], self.lidarpoint[i][2], rot_matrix)
                if d <= self.max_range:
                    for i, (start_angle, end_angle) in enumerate(cones):
                        if start_angle <= a < end_angle:
                            pro = self.calc_proba(self.power[i], d)
                            if pro >= 0.95:
                                self.subsamp.append((self.lidarpoint[i]))





    def file_get(self): #Deze functie zoekt het bestand van de lidar pointcloud die bij de sample hoort. Vervolgens wordt het volledige pad er naar toe gemaakt.
        info =  self.nusc.get('sample', self._sample)
        info_2 = self.nusc.get('sample_data',info['data']['LIDAR_TOP'])
        self.file = os.path.join(self.dataroot, info_2['filename'])

    def lidar_coor(self):

        som = 0
        
        with open(self.file, "rb") as f:
            number = f.read(4)
            self.lidar_punt = 0
            while number != b"":
                quo, rem = divmod(som,5) #omdat je alleen x en y wilt gebruiken en niet de andere dingen kijk je naar het residu van het item waar die op zit.
                if rem == 0: # als het residu = 0 heb je het x coordinaat en res = 1 is het y-coordinaat
                    x = np.frombuffer(number, dtype=np.float32)
                    number = f.read(4) #leest de volgende bit    
                if rem ==1:
                    y = np.frombuffer(number, dtype=np.float32)
                    number = f.read(4) #leest de volgende bit
                if rem ==2:
                    z = np.frombuffer(number, dtype=np.float32)
                    number = f.read(4) #leest de volgende bit
                if rem == 3:
                    intensity = np.frombuffer(number,dtype=np.float32)
                    number = f.read(4) #leest de volgende bit
                if rem == 4:
                    ring_index = np.frombuffer(number,dtype=np.float32)
                    self.lidarpoint.append((x,y,z,intensity,ring_index))
                    self.lidar_punt += 1
                    number = f.read(4)
                else:
                    number = f.read(4) #leest de volgende bit
                som +=1 # som houdt bij hoeveel items gelezen zijn.

    def a_d (self, x, y, z, rot):
        xy = np.array([x,y])
        xy_rot = np.dot(rot,xy)
        angle = math.degrees(math.atan2((xy_rot[1]),(xy_rot[0])))
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
        output_file = f"lidar_data_{self._scene}_{self._sample}.bin"
        with open(output_file, "wb") as bin_file:
            for point in self.lidarpoint:
                point_array = np.array(point, dtype=np.float32)
            	# Write the array as binary data to the file
                bin_file.write(point_array.tobytes())
        return output_file
