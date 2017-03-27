import cv2
import numpy as np
import os

im_counter = 0
path = "./0/"

def crop (im):
    dif = im.shape[1]-im.shape[0]
    print "Dif", dif
    
    im = im [0:im.shape[0], dif/2:(im.shape[1])-dif/2 ]
    print "Shape", im.shape
    return  im
    
    
    
def walk_filter(im,inc=1):
    
    global path
    global im_counter
    if os.path.isdir(path+"Aug"):
        print "Directory  'Aug' already exist"
    else:
        os.mkdir(path+"Aug")
        print "Directory 'Aug' created"
    
    # Get the initial x and y and the final x and y
    xi = yi = 0; xf = yf= im.shape[0]
    #im_counter = 0
    #print "Xi:", xi, "Yf:", yf
    ratio = yf*0.875
    print "Ratio", ratio
    
    while yi+ratio <= yf:
        xi = 0
        while xi+ratio <= xf:
            crp = im[yi:yi+ratio,xi:xi+ratio]
            flip = cv2.flip(crp,1)
            #cv2.imshow("Walker", crp)
            #cv2.imshow("Flip", flip)
            #cv2.waitKey(10)
            xi+=inc
            cv2.imwrite(path+"Aug/"+str(im_counter)+".jpg", crp)
            cv2.imwrite(path+"Aug/"+str(im_counter+1)+".jpg", flip)
            
            im_counter+=2
        yi+=inc
    print "\nLog: %d images write in '%sAug' dir\n" % (im_counter,path)
            

def data_augmentation():
    
    #load image
    global path
    for root, dirs, files in os.walk(path):    
        for f in files:
            if f.endswith(".jpg"):
                im = cv2.imread(path+f)
                
                print "Opening file '%s' in dir '%s'" % (f,os.getcwd())
                print "Original image:", im.shape    
                
                #resize image
                scale = .1
                im = cv2.resize(im,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
                print "resized image:", im.shape        
                
                #cv2.imshow("Original", im)
                
                #crop image
                #cv2.imshow("croped", crop(im))
                im = crop(im)
                #cv2.imshow("croped", im)
                
                #Start image augmentation
                walk_filter(im, 5)
                #cv2.waitKey(0)
        break

def flips():
    im = cv2.imread("./0/2.jpg")
    im1 = cv2.flip(im,0)
    im2 = cv2.flip(im,1)
    im1 = cv2.flip(im1,1)
    cv2.imwrite("13.jpg",im1)
    cv2.waitKey(0)
   


def main():
    for i in range(0,3):
        global path
        path = "./"+str(i)+"/"
        #print path
        data_augmentation()


if __name__ == '__main__':
  main()




