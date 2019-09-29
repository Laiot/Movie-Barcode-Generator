import cv2
import numpy as np
from PIL import Image

src = "1.mov"
target = "test.png"
video = cv2.VideoCapture(src)
FPS = 30
smoothing = 1
stackFactor = 100

def average_frame(img):
    avg_color_per_row = np.average(img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return avg_color


def make_array(video):
    result_array = []

    success, image = video.read()
    counter = 0
    while success:
        if counter%FPS == 0:                                  
            result_array.append(average_frame(image))
        success, image = video.read()                   # image is of type numpy array
        counter += 1

    return np.asarray(result_array, dtype = np.uint8)

# convert video to 1 x #ofFrames array
res_arr = make_array(video)
# rotate to make it horizontal
rotated = np.swapaxes(res_arr,0,1)
# stack the rows
tiled = np.tile([res_arr], (stackFactor, 1, 1))
# blur the image in y-axis
blured = cv2.blur(tiled, (smoothing , 1))
# converti image to rgb from bgr
im_rgb = cv2.cvtColor(blured, cv2.COLOR_BGR2RGB)
# create image from array
im = Image.fromarray(im_rgb, mode='RGB')
# save image
im.save(target)
