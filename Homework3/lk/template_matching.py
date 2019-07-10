import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt

def ssd(img, template):
    k_h, k_w = template.shape
    h, w = img.shape
    res = np.zeros_like(img, dtype=np.float64)
    pad_h, pad_w = int(k_h / 2), int(k_w / 2)

    add_h = 0 if k_h % 2 == 0 else 1
    add_w = 0 if k_w % 2 == 0 else 1

    for i in range(pad_h, h - pad_h):
        for j in range(pad_w, w - pad_w):
            patch = img[i - pad_h:i + pad_h + add_h, j - pad_w:j + pad_w + add_w]
            diff = patch - template
            res[i, j] = np.sum(diff ** 2)

    return res[pad_h:h-pad_h + add_h, pad_w:w-pad_w + add_w]


def ncc(img, template):
    k_h, k_w = template.shape
    h, w = img.shape
    res = np.zeros_like(img, dtype=np.float64)
    pad_h, pad_w = int(k_h / 2), int(k_w / 2)

    add_h = 0 if k_h % 2 == 0 else 1
    add_w = 0 if k_w % 2 == 0 else 1

    template_norm = (template - np.mean(template)) / np.std(template) / (k_h * k_w)
    for i in range(pad_h, h - pad_h):
        for j in range(pad_w, w - pad_w):
            patch = img[i - pad_h:i + pad_h + add_h, j - pad_w:j + pad_w + add_w]
            patch_norm = (patch - np.mean(patch)) / np.std(patch) / (k_h * k_w)
            res[i, j] = np.sum(patch_norm * template_norm)

    return res[pad_h:h - pad_h + add_h, pad_w:w - pad_w + add_w]


def sad(img, template):
    k_h, k_w = template.shape
    h, w = img.shape
    res = np.zeros_like(img, dtype=np.float64)
    pad_h, pad_w = int(k_h / 2), int(k_w / 2)

    add_h = 0 if k_h % 2 == 0 else 1
    add_w = 0 if k_w % 2 == 0 else 1

    for i in range(pad_h, h - pad_h):
        for j in range(pad_w, w - pad_w):
            patch = img[i - pad_h:i + pad_h + add_h, j - pad_w:j + pad_w + add_w]
            res[i, j] = np.sum(np.abs(patch - template))

    return res[pad_h:h - pad_h + add_h, pad_w:w - pad_w + add_w]


if len(sys.argv) != 3:
    print("Specify image path and template path")
    quit()

img_path = sys.argv[1]
templ_path = sys.argv[2]

img = cv2.imread(img_path,0)
img2 = img.copy()
img3 = img.copy()
img4 = img.copy()
template = cv2.imread(templ_path,0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

my_res = ssd(img2, template)
my_min_val, my_max_val, my_min_loc, my_max_loc = cv2.minMaxLoc(my_res)

res_ncc = ncc(img3, template)
min_val_ncc, max_val_ncc, min_loc_ncc, max_loc_ncc = cv2.minMaxLoc(res_ncc)

res_sad = ncc(img4, template)
min_val_sad, max_val_sad, min_loc_sad, max_loc_sad = cv2.minMaxLoc(res_sad)

print(my_res.shape)
print(res.shape)
print(res)
print(my_res)
print(min_val, max_val, min_loc, max_loc)
print(my_min_val, my_max_val, my_min_loc, my_max_loc)

top_left = min_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

cv2.rectangle(img,top_left, bottom_right, 255, 2)
cv2.rectangle(img2,my_min_loc, (my_min_loc[0] + w, my_min_loc[1] + h), 255, 2)
cv2.rectangle(img3,max_loc_ncc, (max_loc_ncc[0] + w, max_loc_ncc[1] + h), 255, 2)
cv2.rectangle(img4,min_loc_sad, (min_loc_sad[0] + w, min_loc_sad[1] + h), 255, 2)

plt.subplot(221),plt.imshow(my_res,cmap = 'gray')
plt.title('Matching Result SSD'), plt.xticks([]), plt.yticks([])

plt.subplot(222),plt.imshow(img2,cmap = 'gray')
plt.title('Detected Point SSD'), plt.xticks([]), plt.yticks([])

plt.subplot(223),plt.imshow(res,cmap = 'gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])

plt.subplot(224),plt.imshow(img,cmap = 'gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

plt.figure(2)
plt.subplot(221),plt.imshow(res_ncc,cmap = 'gray')
plt.title('Matching Result NCC'), plt.xticks([]), plt.yticks([])

plt.subplot(222),plt.imshow(img3,cmap = 'gray')
plt.title('Detected Point NCC'), plt.xticks([]), plt.yticks([])

plt.subplot(223),plt.imshow(res_sad,cmap = 'gray')
plt.title('Matching Result SAD'), plt.xticks([]), plt.yticks([])

plt.subplot(224),plt.imshow(img4,cmap = 'gray')
plt.title('Detected Point SAD'), plt.xticks([]), plt.yticks([])

plt.show()

