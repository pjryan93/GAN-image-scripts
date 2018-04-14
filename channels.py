from scipy.misc import imresize, imsave
from matplotlib.image import imread
import argparse
import os
import numpy as np
 

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dir", required=True,help="dir with files")
	args = vars(ap.parse_args())
	dir_to_load = args["dir"]
	filepaths = []
	
	for dir_, _, files in os.walk(dir_to_load):
		for fileName in files:
			relDir = os.path.relpath(dir_, dir_to_load)
			relFile = os.path.join(relDir, fileName)
			filepaths.append(dir_to_load + "/" + relFile)
	black_and_white = []
	opaq = []
	for i, fp in enumerate(filepaths):
		img = imread(fp)
		if len(img.shape) is not 3:
			print(fp)
			print(img.shape)
			black_and_white.append(fp)
		elif len(img.shape) == 3 and img.shape[-1] != 3:
			print(fp)
			print(img.shape)
			opaq.append(fp)
	for i in black_and_white:
		img = imread(i)
		stacked_img = np.stack((img,)*3,-1)
		imsave( "./" + str(i) + "_conv" + ".png", stacked_img)
	for i in opaq:
		img = imread(i)
		cleaned_img = img[...,:3]
		imsave( "./" + str(i) + "_3chan" + ".png", cleaned_img)
	for i in opaq:
		os.remove(i)
	for i in black_and_white:
		os.remove(i)




