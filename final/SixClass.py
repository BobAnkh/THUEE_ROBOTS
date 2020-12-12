#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Author       : BobAnkh
# @Github       : https://github.com/BobAnkh
# @Date         : 2020-12-09 19:29:53
# @LastEditTime : 2020-12-12 13:01:28
# @Description  : 

# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

from pynq import Overlay
from pynq import Xlnk
import pynq.lib.dma
import numpy as np
from PIL import Image


def img_normalize(img, mean_arg, std_arg):
    r, g, b = img.split()
    r = ((np.array(r) / 255 - mean_arg[0]) / std_arg[0])
    g = ((np.array(g) / 255 - mean_arg[1]) / std_arg[1])
    b = ((np.array(b) / 255 - mean_arg[2]) / std_arg[2])
    return np.array([r, g, b])


def img_crop(img, centerl, centerh, length, height):
    box = (centerl - length / 2, centerh - height / 2, centerl + length / 2,
           centerh + height / 2)
    img = img.crop(box)
    return img


def six_class(orig_img_path):
    overlay = Overlay('./cnnet.bit')
    xlnk = Xlnk()
    x = xlnk.cma_array(shape=(3, 24, 24), dtype=np.float32)
    y = xlnk.cma_array(shape=(6,), dtype=np.float32)
    mean_arg = [0.44087456, 0.39025736, 0.43862119]
    std_arg = [0.18242574, 0.19140723, 0.18536106]
    img = Image.open(orig_img_path)
    height = img.size[1]
    length = img.size[0]
    img = img_crop(img, length/2, height/2, 300, 300)
    img = img.resize((24, 24))
    img = img_normalize(img, mean_arg, std_arg)
    for i in range(24):
        for j in range(24):
            for k in range(3):
                x[k][i][j] = img[k][i][j]
    input_ch = overlay.axi_dma_0.sendchannel
    output_ch = overlay.axi_dma_0.recvchannel
    input_ch.transfer(x)
    input_ch.wait()
    output_ch.transfer(y)
    output_ch.wait()
    result = list(y)
    test_label = ['kabi', 'jienigui', 'miaowazhongzi', 'pikachu', 'yibu', 'xiaohuolong']
    return(test_label[np.argmax(np.array(result))])