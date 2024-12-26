import cv2 as cv
import os

def faceDetection(test_img):
    gray_img = cv.cvtColor(test_img,cv.COLOR_BGR2GRAY)
    face_haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.1,minNeighbors=1)

    return faces,gray_img

def labels_for_training_data(directory):
    faces = []
    faceID=[]

    for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping sysytem file")
                continue