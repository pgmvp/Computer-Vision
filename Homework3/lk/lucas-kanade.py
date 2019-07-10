import numpy as np
import cv2
import os
import sys
from skimage import io
from matplotlib import pyplot as plt

if len(sys.argv) != 2:
    print("Specify path to dataset")
    quit()
data_path = sys.argv[1]

paths = sorted([x for x in os.walk(data_path + 'img/')][0][2])

images = []
for path in paths:
    img = cv2.cvtColor(io.imread(data_path + 'img/' + path), cv2.COLOR_BGR2RGB)
    images.append(img)

init = images[0]

# setup initial location of window
c,r,w,h = None, None, None, None
with open(data_path + 'groundtruth_rect.txt') as file:
    line = file.readline()
    try:
        res = line.split(",")
        c, r, w, h = map(int, res)
    except ValueError:
        res = line.split("\t")
        c, r, w, h = map(int, res)

track_window = (c,r,w,h)

# set up the ROI for tracking
roi = init[r:r+h, c:c+w]
mask = np.zeros(init.shape[:2], dtype=np.uint8)
mask[r:r+h,c:c+w] = 255

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 0,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.05))


# Create some random colors
color = np.random.randint(0,255,(100,3))

# Find corners in ROI
old_gray = cv2.cvtColor(init, cv2.COLOR_RGB2GRAY)
roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = mask, **feature_params)

initc = init.copy()

# Draw keypoints on initial image
for i,p in enumerate(p0):
    x,y = p.ravel()
    initc = cv2.circle(initc,(x,y),5,color[i].tolist(),-1)

# Show initial image
while True:
    initc = cv2.rectangle(initc, (c, r), (c + w, r + h), 255, 2)
    cv2.imshow("Press q to continue", initc)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyWindow("Press q to continue")


for i in range(1, len(images)):
    frame = images[i]
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # recalculate keypoints if there are no good points
    if len(good_new) == 0:
        good_new = cv2.goodFeaturesToTrack(roi_gray, mask = None, **feature_params)

    x,y,w,h = cv2.boundingRect(good_new)
    cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0),2)

    cv2.imshow('frame',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Update the previous frame, previous points and previous roi
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    roi_gray = frame_gray[x:x+w, y:y+h]

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()