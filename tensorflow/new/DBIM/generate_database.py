from PIL import Image
import numpy as np

im = Image.open('input.jpg')
im = (np.array(im))

r = im[:,:,0].flatten()
g = im[:,:,1].flatten()
b = im[:,:,2].flatten()
label = [1]

out = np.array(list(label) + list(r) + list(g) + list(b),np.uint8)
out.tofile("cifar10_data/cifar-10-batches-bin/input.bin")