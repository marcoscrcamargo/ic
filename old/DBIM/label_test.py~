


from PIL import Image
import numpy as np
import os 
import struct

w = 1280 
h = 960
tp = w*h
size = tp*3+1
#print(size)




with open("out.bin", "rb") as f:



    print(struct.unpack('i', f.read(4)))
    label = np.fromfile(f, dtype=np.uint32)
    #print "Read %d" % label
#    print (label)






