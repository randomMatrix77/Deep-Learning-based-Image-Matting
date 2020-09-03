import os
import torch
from torchvision import models
import cv2
import numpy as np
from scipy.spatial import distance

os.environ['TORCH_HOME'] = "D:/Softwares/miniconda/torch_models"
print('current location : {}'.format(os.getenv("TORCH_HOME",os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'),
                                                                         'torch'))))

face_cascade = cv2.CascadeClassifier('haar_face.xml')

path = './user1'
files = os.listdir(path)

det_faces = []
inps = []

for file in files:
    img = cv2.imread(os.path.join(path, file))
    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.resize(bw, (224, 224))
    faces = face_cascade.detectMultiScale(bw)
    try:
        for (x, y, w, h) in faces:
            f = bw[x : x+w, y : y+h]
            f = cv2.resize(f, (224, 224))
            det_faces.append(f)

    except:
        pass

for bw in det_faces:
    bw = bw/255
    bw = np.expand_dims(bw, axis = 0)
    bw = np.vstack([bw]*3)
    bw = np.expand_dims(bw, axis=0)
    inp = torch.tensor(bw, dtype = torch.float32)
    inps.append(inp)

vgg = models.vgg16(pretrained = True)
