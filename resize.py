from scipy.misc import imresize, imsave
from matplotlib.image import imread
import argparse
import os
 

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dir", required=True,help="dir with files")
	ap.add_argument("-o", "--output", required=True,help="path to output directory of images")
	args = vars(ap.parse_args())
	dir_to_load = args["dir"]
	out_dir = args['output']
	filepaths = []
	
	for dir_, _, files in os.walk(dir_to_load):
		for fileName in files:
			relDir = os.path.relpath(dir_, dir_to_load)
			relFile = os.path.join(relDir, fileName)
			filepaths.append(dir_to_load + "/" + relFile)
	for i, fp in enumerate(filepaths):
		img = imread(fp) #/ 255.0
		img = imresize(img, (112, 112))
		imsave(out_dir + "/" + str(i) + ".png", img)