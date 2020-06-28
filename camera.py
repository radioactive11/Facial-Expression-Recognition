import cv2
from model import FacialExpressionModel
import numpy as np

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)


    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        fr = cv2.flip(fr, 1) # correct mirror image effect
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) # convert to grayscale
        faces = facec.detectMultiScale(gray_fr, 1.3, 5) # scale_factor: 1.3, minNeighbours: 5

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            if (pred == "Happy"):
                cv2.putText(fr, pred, (x, y), font, 1, (0, 255, 0), 1)
                cv2.rectangle(fr,(x,y),(x+w,y+h),(0,255,0),1)
            
            elif (pred == "Neutral"):
                cv2.putText(fr, pred, (x, y), font, 1, (0, 255, 255), 1)
                cv2.rectangle(fr,(x,y),(x+w,y+h),(0, 255, 255),1)

            elif (pred == "Angry"):
                cv2.putText(fr, pred, (x, y), font, 1, (0, 255, 0), 1)
                cv2.rectangle(fr,(x,y),(x+w,y+h),(0,255,0),1)
            
            elif (pred == "Sad"):
                cv2.putText(fr, pred, (x, y), font, 1, (255, 0, 0), 1)
                cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0), 1)

            elif (pred == "Fear"):
                cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 255), 1)
                cv2.rectangle(fr,(x,y),(x+w,y+h),(255, 255, 255), 1)

            elif (pred == "Surprise"):
                cv2.putText(fr, pred, (x, y), font, 1, (255, 0, 255), 1)
                cv2.rectangle(fr,(x,y),(x+w,y+h),(255, 0, 255), 1)

            else:
                cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 255), 1)
                cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),1)

            #print(pred)


        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
