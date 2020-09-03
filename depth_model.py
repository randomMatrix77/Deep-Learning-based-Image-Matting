import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ['TORCH_HOME'] = "D:/Softwares/miniconda/torch_models"
print('current location : {}'.format(os.getenv("TORCH_HOME",os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'),
                                                                         'torch'))))

print('Loading Model')
# depth perception model MiDaS by intel
depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS")

# Transform routine for input image
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
# size -> max(384, 384),normalize image

transforms = midas_transforms.default_transform

face_cascade = cv2.CascadeClassifier('haar_face.xml')


def bokeh_filter(path_of_image, blur_kernel_size):

    print('Reading image file')

    if os.path.isfile(path_of_image):
        img = cv2.imread(path_of_image)
    else:
        print('Invalid file')
        return 0

    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    inp = transforms(img)
    out = depth_model(inp)
    out = out / out.max()
    print('Generated depth map')

    bw = cv2.resize(bw, (inp.shape[3], inp.shape[2]))

    print('Detecting faces in image')
    faces = face_cascade.detectMultiScale(bw)
    avg = []

    try:
        for (x, y, w, h) in faces:
            temp = out[0, y:y+h, x:x+h]
            temp = temp.detach().numpy()
            temp_avg = np.average(np.average(temp, axis = 0))
            avg.append(temp_avg)

        avg = np.array(avg)

        clipping_threshold = avg.min()

        print('Face detected. Using {} as clipping value'.format(clipping_threshold))

        # Resize output to original (image) size
        out = torch.nn.functional.interpolate(out.unsqueeze(1), size=img.shape[:2],
                                            mode="bicubic", align_corners=False).squeeze()
        out = out.detach().numpy()

        print('Extracting bg and fg.')
        # Foreground & Background extraction
        fg = (out > clipping_threshold)
        bg = 1 - fg

        fg_im = img * fg[:,:,np.newaxis]    # foreground image of shape (x, y, 3)

        print('Applying Gaussian Blur')
        blur = cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0)
        blur = blur*bg[:, :, np.newaxis]

        bokeh = blur + fg_im

        return bokeh

    except:
        print('No faces found. Bokeh filter works only if faces are detected.')
        return 0
