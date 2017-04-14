# import the necessary packages
from sklearn.neural_network import MLPClassifier
# from sklearn.cross_validation import train_test_split
# resolvendo problemas de compatibilidade
from sklearn.model_selection import train_test_split

from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os

def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins=(8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
 
	# handle normalizing the histogram if we are using OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)
 
	# otherwise, perform "in place" normalization in OpenCV 3 (I
	# personally hate the way this is done
	else:
		cv2.normalize(hist, hist)
 
	# return the flattened histogram as the feature vector
	return hist.flatten()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=5,
	help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs for k-NN distance (-1 uses all available cores)")
ap.add_argument("-c", "--classify", default="",
	help="path to input image")

args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images(args["dataset"]))
 
# initialize the raw pixel intensities matrix, the features matrix,
# and labels list
rawImages = []
features = []
labels = []

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
	# load the image and extract the class label (assuming that our
	# path as the format: /path/to/dataset/{class}/{image_num}.jpg
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[1]
 	
	# extract raw pixel intensity "features", followed by a color
	# histogram to characterize the color distribution of the pixels
	# in the image
	pixels = image_to_feature_vector(image)
	hist = extract_color_histogram(image)
 
	# update the raw images, features, and labels matricies,
	# respectively
	rawImages.append(pixels)
	features.append(hist)
	labels.append(label)
 
	# show an update every 1,000 images
	if i > 0 and i % 1000 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))

# show some information on the memory consumed by the raw images
# matrix and features matrix
rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)
print("[INFO] pixels matrix: {:.2f}MB".format(
	rawImages.nbytes / (1024 * 1000.0)))
print("[INFO] features matrix: {:.2f}MB".format(
	features.nbytes / (1024 * 1000.0)))



img_path = args["classify"]
if(img_path != ""):
	# partition the data into training and testing splits, using 75%
	# of the data for training and the remaining 25% for testing
	(trainRI, testRI, trainRL, testRL) = train_test_split(
		rawImages, labels, test_size=0, random_state=42)
	(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
		features, labels, test_size=0, random_state=42)

	
	img = cv2.imread(img_path)
	pxl = image_to_feature_vector(np.array(img)).reshape(1,-1)
	hst = extract_color_histogram(np.array(img)).reshape(1,-1)


	model = MLPClassifier(hidden_layer_sizes=(100,100,100),
			solver='sgd',learning_rate_init=0.01, 
			max_iter=500)

	model.fit(trainRI, trainRL)
	
	print("(pixels) image label:" + model.predict(pxl)[0])

	model = MLPClassifier(hidden_layer_sizes=(100,100,100),
			solver='sgd',learning_rate_init=0.01, 
			max_iter=500)

	model.fit(trainFeat, trainLabels)

	print("(histogram) image label:" + model.predict(hst)[0])

else:
	# partition the data into training and testing splits, using 75%
	# of the data for training and the remaining 25% for testing
	(trainRI, testRI, trainRL, testRL) = train_test_split(
		rawImages, labels, test_size=0.25, random_state=42)
	(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
		features, labels, test_size=0.25, random_state=42)



	# train and evaluate a k-NN classifer on the raw pixel intensities
	print("[INFO] evaluating raw pixel accuracy...")
	model = MLPClassifier(hidden_layer_sizes=(100,100,100),
			solver='lbfgs',learning_rate_init=0.01, 
			max_iter=500)

	model.fit(trainRI, trainRL)
	acc = model.score(testRI, testRL)
	print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))



	# train and evaluate a k-NN classifer on the histogram
	# representations
	print("[INFO] evaluating histogram accuracy...")
	model = MLPClassifier(hidden_layer_sizes=(100,100,100),
			solver='sgd',learning_rate_init=0.01, 
			max_iter=500)
	model.fit(trainFeat, trainLabels)
	acc = model.score(testFeat, testLabels)
	print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))

