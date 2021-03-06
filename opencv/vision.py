from classifiers import svm, knn, mlp

import numpy as np


def initializate(classifier='knn'):
	if(classifier == 'knn'):
		knn.fit()
	elif(classifier == 'mlp'):
		mlp.fit()
	elif(classifier == 'svm'):
		svm.fit()

def classify(img_path, classifier='knn'):
	if(classifier == 'knn'):
		return knn.classify(img_path)
	elif(classifier == 'mlp'):
		return mlp.classify(img_path)
	elif(classifier == 'svm'):
		return svm.classify(img_path)

def print_proba(ret, classifier='knn', full=False):
	if(classifier == 'knn'):
		knn.print_proba(ret, full)
	elif(classifier == 'mlp'):
		mlp.print_proba(ret, full)
	elif(classifier == 'svm'):
		svm.print_proba(ret, full)
