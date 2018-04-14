from scipy.misc import imresize, imsave
from matplotlib.image import imread
import argparse
import os
import numpy as np

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-u", "--urls", required=True,help="file with urls")
	ap.add_argument("-o", "--outfile", required=True,help="file to write to")
	args = vars(ap.parse_args())
	fileName = args['urls']
	outFile = args['outfile']
	with open(fileName,'r') as f:
		content =f.readlines()
		cleanest  = list(set(content))
		with open(outFile,'w') as o:
			for i in cleanest:
				o.write(i)




