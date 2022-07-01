import os
import numpy as np
from PIL import Image
import cv2
import pickle

BASE_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_dir,"images")
face_cascade = cv2.CascadeClassifier('clasificatori/data/haarcascade_frontalface_alt2.xml')
# eye_cascade = cv2.CascadeClassifier('clasificatori/data/haarcascade_eye.xml')
recognition = cv2.face.LBPHFaceRecognizer_create()
# print(BASE_dir, image_dir)

current_id = 0
label_ids = {}
x_train = []
y_labels = []


for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ","-").lower()
            # print(label, path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
        id_ = label_ids[label]
        print(label_ids)
        pil_image = Image.open(path).convert('L')
        size = (550, 550)
        final_image = pil_image.resize(size,Image.Resampling.LANCZOS)
        image_array = np.array(pil_image, 'uint8')
        # print(image_array)

        faces =face_cascade.detectMultiScale(image_array, scaleFactor = 1.3, minNeighbors =  5)
        for (x, y, w, h) in faces:
            print(x, y, w, h)
            roi = image_array[y: y+h, x:x+w]
            x_train.append(roi)
            y_labels.append(id_)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids,f)


recognition.train(x_train, np.array(y_labels))
recognition.save("trainer.yml")



