import matplotlib.pyplot as plt
import numpy as np
import os
import time

import tensorflow as tf

import os
import json

map_matrix = {
    "0":[],
    "1":[],
    "2":[],
    "3":[],
    "4":[],
    "5":[],
    "6":[],
    "7":[],
    "8":[],
    "9":[],
    "10":[],
    "11":[],
    "12":[],
    "13":[],
    "14":[],
    "15":[],
    "16":[],
    "17":[],
    "18":[],
    "19":[],
    "20":[],
    "21":[],
    "22":[],
    "23":[]
}

paths = list(os.scandir("./out"))

# Filling the matrix

AUGMENT_MUL = 16

for j in range(len(paths)):
    if paths[j].is_file():
        file = open('out/'+paths[j].name, 'r')
        data = json.loads(file.readline())
        tileset = str(data['tileset'])
        map_matrix[tileset].append([])
        width = data['width']
        height = data['height']
        pos = len(map_matrix[tileset]) - 1
        for i in range(0,data['height']):
            map_matrix[tileset][pos].append([])
            for k in range(0,width):
                byteval = data['map_data'][i*width+k]
                #print(byteval*AUGMENT_MUL)
                map_matrix[tileset][pos][i].append(byteval)
        for augm_counter in range(2,AUGMENT_MUL+1):
                    if not os.path.exists('out/'+paths[j].name[0:-5]+'_augm_'+str(augm_counter)+'.json'):
                        file = open('out/'+paths[j].name[0:-5]+'_augm_'+str(augm_counter)+'.json', 'w')
                        file.write(json.dumps({
                            'height': height,
                            'width': width*augm_counter,
                            'map_data': [val  for row in map_matrix[tileset][pos] for elem in row for val in [elem]*augm_counter],
                            'tileset':tileset
                        })
                    )
        """
        print(data)
        print(rotated_180)
        print(map_matrix[tileset][pos])
"""
