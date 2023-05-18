from sys import argv

import numpy as np
import json

offsets = []

romfile = open("rom.gb","rb") 

romfile.seek(0x01AE, 0)
pointer_bytes = np.array_split(list(romfile.read(494)),247)

romfile.seek(0xC23D, 0)
banks_bytes = np.array_split(list(romfile.read(247)), 247)

pointers = list(map(lambda val: int.from_bytes(list(val),"little") % 0x4000, pointer_bytes))
banks = list(map(lambda val: int.from_bytes(list(val),"little"), banks_bytes))

banks_map = {}

width = 0
height = 0
pointer_map = []

for i in range(0, 247):
    offset = pointers[i] + 0x4000 * banks[i]
    offsets.append(offset)
    banks_map[offset] = banks[i]
#print(banks_map[0x182A1])

for offset in offsets:
    if(offset != 0x45CE5 and offset != 0x56B2 and offset != 0x762A2 and offset != 0x49A4 and offset != 0x5704):
        romfile.seek(offset, 0)
        tileset = int.from_bytes(list(romfile.read(1)),"little")
        height = int.from_bytes(list(romfile.read(1)),"little")
        width = int.from_bytes(list(romfile.read(1)),"little")
        pointer_map = int.from_bytes(list(romfile.read(2)),"little")
        abs_pointer_map = pointer_map % 0x4000 + 0x4000 * banks_map[offset]
        if(offset == 0x182A1):
            print(hex(abs_pointer_map))
            print(hex(pointer_map))

        if(abs_pointer_map != 0):
            datafile = open('out/'+str(hex(offset))+'.json','w')
            romfile.seek(abs_pointer_map, 0)
            map_data = list(romfile.read(width * height))
            data = {
                "width":width,
                "height":height,
                "map_data":map_data,
                "tileset":tileset
            }

            datafile.write(json.dumps(data))
            datafile.close()

romfile.close()


