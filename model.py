# MIT License

# Copyright (c) 2019 Runway AI, Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
#Tools lib
import numpy as np
import cv2
import random
import time
import os
import argparse
#Models lib
from models import *
#Metrics lib

class DERAINDROP_MODEL():

    def __init__(self, opts):
        self.model = Generator().cuda()
        self.model.load_state_dict(torch.load('./weights/gen.pkl'))
        print('done.')
        
    # Generate an image based on some text.
    def clean(self, img):
        print('starting inference...')
        img = self.align_to_four(np.array(img))
        result = self.predict(img)
        print('done.')
        return np.uint8(result)

    def align_to_four(self, img):

        a_row = int(img.shape[0]/4)*4
        a_col = int(img.shape[1]/4)*4
        img = img[0:a_row, 0:a_col]
        return img


    def predict(self, image):
        image = np.array(image, dtype='float32')/255.
        image = image.transpose((2, 0, 1))
        image = image[np.newaxis, :, :, :]
        image = torch.from_numpy(image)
        image = Variable(image).cuda()

        out = self.model(image)[-1]

        out = out.cpu().data
        out = out.numpy()
        out = out.transpose((0, 2, 3, 1))
        out = out[0, :, :, :]*255.
        
        return out   