import train_utils
import cv2
import os
import numpy as np

if __name__ == "__main__":
    filenames = []
    for f in os.listdir('frames/content/'):
        if f.startswith('frame'):
            filenames.append(f)
    # sort files
    filenames = sorted(filenames, key=lambda f: int(f.strip('frame_*.png')))
    print(filenames)
    images = []
    for f in filenames:
        img = cv2.imread('frames/content/'+f, 0)
        images.append(img)
    print(np.array(images).shape)
    train_utils.make_video(np.array(images), 'video')

    #train_utils.make_video(images, "video")