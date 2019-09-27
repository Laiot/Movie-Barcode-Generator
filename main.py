import sys
import cv2
import numpy as np
from PIL import Image
from multiprocessing import Pool
import os

os.system("taskset -p 0xff %d" % os.getpid())


def frame_avg(img):
    scaled = img.astype('uint32')
    squared = scaled**2
    avgsq = np.average(squared, axis=1)
    return np.sqrt(avgsq).astype('uint8')

def movie_iter(movie_name, frames_to_skip):
    movie=cv2.VideoCapture(f"{movie_name}.mp4") #
    s, f = movie.read()
    while s:
        yield f
        for i in range(frames_to_skip):
            movie.read()
        s,f = movie.read()


def elab(movie_it):
    with Pool(8) as p:
        res = p.imap(frame_avg, movie_it, chunksize=10)
        res = list(res) 
    #res = list(map(frame_avg, movie_it))

    return res

if __name__=='__main__':
    it = list(movie_iter(sys.argv[1], 4))
    res = elab(it)
    c = np.array(res)
    cc = c.swapaxes(0,1)
    Image.fromarray(cc, mode='RGB').save(f"{sys.argv[1]}.jpg")

