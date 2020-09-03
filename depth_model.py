import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ['TORCH_HOME'] = "D:/Softwares/miniconda/torch_models"
print('current location : {}'.format(os.getenv("TORCH_HOME",os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'),
                                                                         'torch'))))

# depth perception model MiDaS by intel
depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS")

# Transform routine for input image
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
# size -> max(384, 384),normalize image

transforms = midas_transforms.default_transform

img = cv2.imread("D:\Personal\mohsinnnnn\imp stuff\Snow\IMG_20161108_124416.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
inp = transforms(img)
out = depth_model(inp)

# Resize output to original (image) size
out = torch.nn.functional.interpolate(out.unsqueeze(1), size=img.shape[:2],
                                    mode="bicubic", align_corners=False).squeeze()
out = out.detach().numpy()

plt.imshow(out)

# image matting part

face_cascade = cv2.CascadeClassifier('haar_face.xml')

# detect face in the image
bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(bw)

for (x, y, w, h) in faces:
     f = out[x : x+h, y : y+h]

avg = np.average(np.average(f, axis = 0))

# Foreground & Background extraction
fg = (out > avg)
bg = 1 - fg

fg_im = img * fg[:,:,np.newaxis]    # foreground image of shape (x, y, 3)

blur = img*bg[:, :, np.newaxis]
blur = cv2.GaussianBlur(blur, (13, 13), 0)

bokeah = blur + fg_im
