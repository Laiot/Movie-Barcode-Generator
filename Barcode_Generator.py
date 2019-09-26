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
        res = p.map(frame_avg, movie_it, chunksize=100)

    return res

complete = collect_frames(movie)

c = np.array(complete)

cc = c.swapaxes(0, 1)

i = Image.fromarray(cc, mode='RGB')

i.save('multip_test_steven.jpg')
