Scripts meanshift.py, camshift.py, lucas-kanade.py and lucas-kanade-pyr.py need to be launched with command line argument specifying the path to the dataset with video frames.
Datasets were taken from here: http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html.
Please, specify path to the parent folder itself (e. g. /home/CV/Surfer/), because script seeks for groundtruth.txt file to obtain the initial ROI.

Meanshift, Camshift and Lucas-Kanade were tested on Trellis, Surfer, BlurFace, DragonBaby, Panda, Biker, Basketball, Bird1, BlurCar2 datasets.

Script template_matching.py needs to be launched with arguments specifying the path to the image and to the template.
