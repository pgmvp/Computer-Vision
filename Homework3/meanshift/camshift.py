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

while True:
    img2 = cv2.rectangle(init.copy(), (c, r), (c + w, r + h), 255, 2)
    cv2.imshow("Press q to continue", img2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyWindow("Press q to continue")

# set up the ROI for tracking
roi = init[r:r+h, c:c+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# compute hue histogram of the ROI and visualize it
roi_hist = cv2.calcHist([hsv_roi],[0],None,[180],[0,180])
plt.plot(roi_hist)

# determine the most occuring hue in ROI
argmax = float(np.argmax(roi_hist))

# threshold image to obtain only colors which are close to the most occuring one
mask = cv2.inRange(hsv_roi, np.array([max(argmax-15,0.), 60.,50.]), np.array([min(argmax+15, 180.),255.,255.]))

roi_hist2 = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
# cv2.normalize(roi_hist2, roi_hist2,0,255,cv2.NORM_MINMAX)
plt.figure(2)
plt.plot(roi_hist2)
plt.show()


# Setup the termination criteria, either 10 iterations or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

for i in range(1, len(images)):
    frame = images[i]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist2, [0, 180], 1)
    cv2.imshow("Prob",dst)

    # apply camshift to get the new location
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)

    # Draw it on image
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    img2 = cv2.polylines(frame, [pts], True, 255, 2)
    cv2.imshow('img2', img2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

