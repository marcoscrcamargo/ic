from PIL import Image
import numpy as np
import os 
import random as rd

#im=list()
w = 84 
h = 84
tp = w*h
size = tp*3+1
#print(size)

#out = np.array(0+0+0+0)

#fnumb = 0
#out = np.zeros(1,1)
#cur = 0

def walk_dir():

    for dirpath, dnames, fnames in os.walk("./"):
        for f in fnames:
            #print "Len of fnames", len(fnames)
            #print
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
                    out = np.array(aux)
                    #print "Shape aux:", aux.shape, "out", out[0,] 
                else:
                    #out = np.concatenate((out,aux))
                    
                    
                     
                    out = np.vstack([out,aux])
                    
                    print "Shape out", out.shape 
                    
                    
                    #out += aux 
                               
                #print "Size out file: %.2lf MB" % (len(out)/1000000)
    return out               
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



def main():
    os.system('clear')
    
    out = walk_dir() 
    '''
    out = np.array([range(0,100)])
    for i in range(0,100):
        vec = np.full( (1,100), i, dtype=int )
        out = np.vstack([out,vec])    
    '''
    
    print "\n\n\n***SUMARY:***\n\nOut shape:", out.shape
    #number of classes
    nclass = 3

    #number of images
    nim = len(out)
    print "Number of images:", nim

    #test dataset size percent
    ds_per= 0.1

    #test dataset size
    ts= int(nim*ds_per)
    print "Number of test set:", ts
        
    index = rd.sample(range(1,len(out)),ts) 
    #print "Chosen index : ", index
    
    #for i in range(0,len(index)): index[i]=index[i]-1
    #print "Chosen index: ", index
    
    #get the test set in the array out
    testset = out[index, ]
    #print "Test Set: ", testset
    

    #print "OUT len:", len(out)
     
    trainIndex = range(0,len(out))
    
    #training indexes
    ti = [x for x in trainIndex if x not in index]
    
    #print "index of train ", len(ti)
    
    
    trainset = out[ti,]
    
   
    
    print "Train Set lenght:" , len(trainset)
    
    print "Total size Train + Test:", len(trainset)+len(testset)
    
    
    
    path = '/home/dtozadore/Proj_Python/cifar10_data/cifar-10-batches-bin/'
    
    trainset.tofile(path + "trainset.bin")    
    trainset.tofile(path + "testset.bin")    
    
    print 
    print "Files 'trainset.bin' and 'testset.bin' write in path: "
    print path
    
    print "\n\n\nEND."
    
    
if __name__ == '__main__':
  main()




        
