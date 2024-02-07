import cv2
import numpy as np
import os
from PIL import Image
import imageio
import matplotlib.pyplot as pyplot


img_path = './jpg_img'
raw_path = './raw_ir'
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
    #     raw_data = np.frombuffer(raw_file.read(), dtype=np.uint8)
    #
    # # 根据RAW数据创建图像
    # image = Image.frombuffer('I;16', imgshape, raw_data, 'raw', 'I;16', 0, 1)
    # image = image.convert('RGB')
    # # image.show()
    # # 保存为JPEG
    # image.save(new_path, 'JPEG')

    raw_image = np.fromfile(path, dtype=np.uint16)  # 假设RAW图像的文件名为input.raw

    # 2. 根据图像尺寸调整形状（假设图像宽度为width，高度为height）
    width = 640  # 替换为实际图像的宽度
    height = 480  # 替换为实际图像的高度
    raw_image = raw_image.reshape((height, width))

    # 3. 对IR数据进行处理（根据实际需求）
    # 在这里，你可能需要对raw_image进行一些预处理，例如归一化、拉伸等，具体取决于你的IR图像的特性。

    # 4. 将IR图像转换为8位的灰度图像
    ir_image = cv2.normalize(raw_image, None, 0, 255, cv2.NORM_MINMAX)
    ir_image = ir_image.astype(np.uint8)
    cv2.imwrite(new_path, ir_image)
