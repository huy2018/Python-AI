import cv2
import os
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, 'image')

face_cascades = cv2.CascadeClassifier('cascades/data/haarcascades/haarcascade_frontalface_alt2.xml')




recognizer = cv2.face.LBPHFaceRecognizer_create()


y_label = []
x_train = []
current_id = 0
label_id = {}

for root, dirs , files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith('jpg'):
            path = os.path.join(root, file)
            label = os.path.basename(root)
            #print(label, path)
            if not label in label_id:
                label_id[label] = current_id
                current_id += 1
            id =label_id[label]
            #print(label_id)

            pil_image = Image.open(path).convert("L") # grayScale
            image_array = np.array(pil_image, "uint8")
            #print(image_array)
            faces = face_cascades.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            for (x, y, w , h) in faces:
                roi = image_array[y:y+h, x:x+w]
            x_train.append(roi)
            y_label.append(id)


#print(x_train)
#print(y_label)

with open("labels.pickle", "wb") as f:
    pickle.dump(label_id, f)

recognizer.train(x_train, np.array(y_label))
recognizer.save("trainner.yml")
