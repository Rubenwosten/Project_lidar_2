
from nuscenes.nuscenes import NuScenes 
from Cell import Cell

import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
from Severity import severity


from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper

from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer

from nuscenes.prediction.models.backbone import ResNetBackbone
from nuscenes.prediction.models.mtp import MTP
from nuscenes.prediction.models.covernet import CoverNet
import torch

num_of_modes = 5
lengte = 24


class Object:
    def __init__(self, map, constant_power):
        #data object
        self._sample = None
        self._sampleindex = None
        self._x= None
        self._y = None
        self.oud = None
        self.map = map
        self.xmin =map.patch[0]
        self.ymin =map.patch[2]
        self.anns = None
        self.width = self.map.grid.width
        self.length = self.map.grid.length

        self.constant_power = constant_power
        #nusc function
        self.reso= map.grid.res
        self.nusc = map.nusc
        self.nusc_map = map.nusc_map
        self.helper = helper = PredictHelper(self.nusc)
        #Prediction function
        self.static_layer_rasterizer = StaticLayerRasterizer(helper)
        self.agent_rasterizer = AgentBoxesWithFadedHistory(helper, seconds_of_history=1)
        self.mtp_input_representation = InputRepresentation(self.static_layer_rasterizer, self.agent_rasterizer, Rasterizer())
        self.backbone = ResNetBackbone('resnet50')
        self.mtp = MTP(self.backbone, num_modes=num_of_modes)
        
    @property
    def sample(self):
        return self._sample
    
    #@sample.setter
    #def sample(self, values):
        

    def update(self, sample, x, y, sample_index, object_list_new=[], prnt=False):
        # add a check whether the tracking risk has already been set
        self._sample = sample
        self._x = x
        self._y = y
        self._sampleindex = sample_index
        total_sev = 0
        if self._sample != self.oud:
            if self.constant_power == True:
                info = self.nusc.get('sample', self._sample)
                anns = info['anns']
                print(len(anns))
            else: 
                anns = object_list_new
                #print(anns)

            if prnt:
                print(f'amount of objects within the sample = {len(anns)}')
            
            for i in range(len(anns)):
                ans = anns[i]
                
                info = self.nusc.get('sample_annotation', ans)
                rot = np.arctan2((2*(info['rotation'][0]*info['rotation'][3]+info['rotation'][1]*info['rotation'][2])),(1-2*(info['rotation'][3]**2+info['rotation'][2]**2)))
                voor = self.voorspelling(info['instance_token'])
                sev = severity.factor(info['category_name'], info['rotation'], info['translation'], self.nusc.ego_pose[i]['rotation'], self._x, self._y, detected=True)
                #print(f'sev = {sev}')
                total_sev += sev
                gespl , prob = self.route_splitser(num_of_modes,lengte, voor)
                j=0
                if np.isnan(gespl).any():
                    continue
                else:
                    for j in range(num_of_modes):
                            box = self.bounding_box(info['size'], rot, int(gespl[2*j][0] + info['translation'][0]), int(gespl[2*j+1][0]+ info['translation'][1]))
                            self.risk_to_cell(box, prob, j, sev)
                            if prnt:
                                print("1 num of modes klaar")
                                print (box)
                            continue
                    continue
            self.oud = self._sample
            return len(anns), total_sev
        else: 
             return len(anns), total_sev

    def voorspelling(self,objecttoken):
        img = self.mtp_input_representation.make_input_representation(objecttoken,self._sample)
        agent_state_vector = torch.Tensor([[self.helper.get_velocity_for_agent(objecttoken, self._sample),
                                    self.helper.get_acceleration_for_agent(objecttoken, self._sample),
                                    self.helper.get_heading_change_rate_for_agent(objecttoken, self._sample)]])
        image_tensor = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0)
        voorspelling = self.mtp(image_tensor, agent_state_vector)
        return voorspelling
    
    def route_splitser(self, num_of_modes,route_length,voorspell):
        route_dim = (num_of_modes*2,route_length/2)
        route_tensor = voorspell.flatten()
        routestensor = route_tensor[:num_of_modes * route_length].view(-1)
        probabilities_tensor = route_tensor[num_of_modes * route_length:]
        
        gespilts = routestensor.view(num_of_modes, 2, -1).permute(1, 2, 0).reshape(int(route_dim[0]), int(route_dim[1])).detach().numpy()
        prob_logit = probabilities_tensor.detach().numpy()
        som = 0
        i = 0
        for i in range(num_of_modes):
            som += math.exp(prob_logit[i])
            i+=1
        i=0
        prob = np.empty(num_of_modes)
        for i in range(num_of_modes):
            prob[i] = math.exp(prob_logit[i])/som
            i +=1
        return gespilts, prob

    def risk_to_cell(self, box,prob, i, sev):
                j = np.min(box[:,0])
                while j < np.max(box[:,0]):
                    k = np.min(box[:,1])
                    while k < np.max(box[:,1]):
                        if (int((j-self.xmin)/self.reso)<0 or int((k-self.ymin)/self.reso)<0 or int((j-self.xmin)/self.reso)>=self.width or int((k-self.ymin)/self.reso)>=self.length):
                            k+=self.reso
                        else:
                            self.map.grid.get_cell(int((j-self.xmin)/self.reso),int((k-self.ymin)/self.reso)).track_risk[self._sampleindex]+=prob[i] * sev
                            #print(f'prob[i] * sev = {prob[i] * sev}')
                            k+=self.reso
                    j+=self.reso
    
    def bounding_box(self, size, rotation, x, y):
        box = np.array([
            [-0.5 * size[0], -0.5 * size[1]],
            [0.5 * size[0], -0.5 * size[1]],
            [-0.5 * size[0], 0.5 * size[1]],
            [0.5 * size[0], 0.5 * size[1]]
            ])
        rot = rot = np.array([
            [np.cos(rotation), -np.sin(rotation)],
            [np.sin(rotation), np.cos(rotation)]
        ])
        rotbox = np.dot(rot,box.T).T
        rotbox = rotbox + np.array([x,y])
        return rotbox
    
