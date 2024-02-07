import cv2 as cv
import os, pickle

data = pickle.loads(open(os.getcwd() + '\\encodings_webcam_uni.pickle', 'rb').read())

print (data)
#img = cv.imread('E:\\issac\\img1.jpg')
#new_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#cv.imshow("new_rgb",new_rgb)

#cv.waitKey(0)