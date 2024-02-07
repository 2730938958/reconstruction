import cv2
import numpy as np
import os
from PIL import Image
import imageio
import matplotlib.pyplot as pyplot


img_path = './jpg_img'
raw_path = './raw_depth'
imgshape = (640, 480)
raws = os.listdir(raw_path)
for raw in raws:
    path = raw_path + '\\' + raw
    new_path = img_path+'\\'+raw.split('.')[-2]+'.png'

    # rawfile = np.fromfile(path, dtype=np.uint16)  # 以float32读图片
    # print(rawfile.shape)
    # rawfile.shape = imgshape
    # print(rawfile.shape)
    # b = rawfile.astype(np.uint8)  # 变量类型转换，float32转化为int8
    # print(b.dtype)
    # pyplot.imshow(rawfile)
    # imageio.imwrite(new_path, b)

    # with open(path, 'rb') as raw_file:
    #     raw_data = np.frombuffer(raw_file.read(), dtype=np.uint16)
    #
    # # 根据RAW数据创建图像
    # image = Image.frombuffer('I;16', imgshape, raw_data, 'raw', 'I;16', 0, 1)
    # image = image.convert('RGB')
    # # image.show()
    # # 保存为JPEG
    # image.save(new_path, 'JPEG')

    depth_data = np.fromfile(path, dtype=np.uint16)
    depth_image = depth_data.reshape((480, 640))
    depth_image_scaled = (depth_image / np.max(depth_image) * 255).astype(np.uint8)
    # 保存深度图像为PNG
    cv2.imwrite(new_path, depth_image_scaled)
