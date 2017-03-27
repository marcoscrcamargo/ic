from PIL import Image
import numpy as np
import os 

#im=list()
w = 84 
h = 84
tp = w*h
size = tp*3+1
#print(size)

#out = np.array(0+0+0+0)

fnumb = 0

for dirpath, dnames, fnames in os.walk("./"):
    for f in fnames:
        print "Len of fnames", len(fnames)
        print
        if f.endswith(".jpg"):
            #os.system('clear')
            #print("Dir:", d in dnames, "File: ",f )
            print "File '%s' in path '%s' " % (f,dirpath)
            #print "Current path: %s" % dirpath 
            im = Image.open(dirpath+"/"+f)
            im = (np.array(im))

            r = im[:,:,0].flatten()
            g = im[:,:,1].flatten()
            b = im[:,:,2].flatten()
            foo = int(dirpath[len(dirpath)-1])
            #print "Label: %d" % foo
            label = [foo]
            
            aux = np.array(list(label) + list(r) + list(g) + list(b),np.uint8)
            #print "aux: ", len(aux), "should be ", size
            
           
            try:
                out
            except:
                out = aux
            else:
                out = np.concatenate((out,aux)) 
                #out += aux 
                           
            #print "Size out file: %.2lf MB" % (len(out)/1000000)
            print #(out)
    '''
    try:
        out
    except:
        print "Out not declared yet"
    else:
       out = np.array(out, dtype=np.uint8)  
       out.tofile("out%d.bin" % fnumb)
       fnumb+=1
       del out          
    #'''

#for i in range(0,300):
   # print "Label in out: ", out[(50+i)*size]    
    
#out = np.array(out, dtype=np.uint8)  
print 
print "numer of images:", len(out)/size

#number of classes
nclass = 3

#number of images
nim = len(out)/size

#test dataset size percent
ds_per= 0.1

#test dataset size
ts= nim*ds_per









#out.tofile("out.bin")    






    
