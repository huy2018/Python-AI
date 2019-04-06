import numpy as np
import cv2
import pickle


my_res ='720p'

# Set resolution for the video capture
def change_res (cap,width ,height):
    cap.set(3,width)
    cap.set(4,height)

# Standard Video Dimensions Sizes
STD_DIMENSIONS = {
    "480p": (640,480),
    "720p": (1280,720),
    "1080p": (1920,1080),
    "4k": (3840,2160),
}
# grab resolution dimensions and set video capture to it.
def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width,height = STD_DIMENSIONS[res]
    ## change the current caputre device
    ## to the resulting resolution
    change_res(cap, width, height)
    return width, height

cap = cv2.VideoCapture(0)

dims = get_dims(cap, res=my_res)

face_cascades = cv2.CascadeClassifier('cascades/data/haarcascades/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascades/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascades/haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()

labels = {'person_name': 1}
with open("labels.pickle", "rb") as f:
    og_labels= pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

recognizer.read("trainner.yml")
while True:
    # Capture Frame-by frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascades.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        #print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]  # (ycord_start , ycord_end)
        roi_color = frame[y:y+h, x:x+w]

        #img_item = "my_image.png"
        #cv2.equalizeHist('my_image.png')
        #cv2.imwrite(img_item, roi_color)

        id_, conf = recognizer.predict(roi_gray)
        if conf>= 80:# 5 #and conf <= 85:
            #print(id_)
            #print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 0, 0)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)


        color = (0, 255, 0)  # B G R
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

        # smile interface
        #subitems = smile_cascade.detectMultiScale(roi_gray)
        #for (ex, ey, ew, eh) in subitems:
           # cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Display The Resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

