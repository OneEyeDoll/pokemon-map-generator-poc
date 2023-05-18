from sys import argv

import numpy as np
import json

PALLET_TOWN_OFFSET = 0x182A1
PALLET_TOWN_BANK = 6

PALLET_TOWN_OBJ_OFFSET = 0x42FD 

romfile = open("rom1.gb","r+b") 

romfile.seek(PALLET_TOWN_OFFSET + 1, 0)

#height
romfile.write(bytes([9]))
#width
romfile.write(bytes([52]))
abs_pointer = 0x182FD#0x1C2FD#0x4000 * PALLET_TOWN_BANK + PALLET_TOWN_OBJ_OFFSET

datafile = open("./out/0x1a19d_augm.json","r") #./tfout/final.json
map_data = json.loads(datafile.readline())
datafile.close()

town_to_push = map_data['map_data']
print(town_to_push)
romfile.seek(abs_pointer, 0)
for row in town_to_push:
    #for asset in row:
        #print(int(str(asset).encode()))
        romfile.write(bytes([row])) 
romfile.close()


