import cv2
import numpy as np
from PIL import Image

movie = cv2.VideoCapture("movie.mp4") # VideoCapture take as argument any video files, image sequences or cameras


def frame_avg(img):
    scaled = img.astype('uint32')
    squared = scaled**2
    avgsq = np.average(squared, axis=0)
    return np.sqrt(avgsq).astype('uint8')


def collect_frames(movie):
    res = []
    s, i = movie.read()
    while s:
        res.append(frame_avg(i))
        s, i = movie.read()
        for i in range(30):
            if s:
                s, i = movie.read()
    return res


complete = collect_frames(movie)

c = np.array(complete)

cc = c.swapaxes(0, 1)

i = Image.fromarray(cc, mode='RGB')

i.save('barcode.jpg') # Name of your output
