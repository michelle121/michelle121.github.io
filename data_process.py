#数据处理
import os
import cv2
import numpy as np #支持大量维度的数组与矩阵运算
from tqdm import tqdm #进度条包
import argparse

from utils import Preprocess

#创建解析对象parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='photo folder path')
parser.add_argument('--save_path', type=str, help='save folder path')

args = parser.parse_args()
os.makedirs(args.save_path, exist_ok=True)

#数据预处理
pre = Preprocess()

#enumerate枚举：
#os.listdir():返回指定文件夹包含的文件或文件夹的名字的列表
for idx, img_name in enumerate(tqdm(os.listdir(args.data_path))):
    # cv2.cvtColor()颜色转换
    #os.path.join：拼接文件路径
    img = cv2.cvtColor(cv2.imread(os.path.join(args.data_path, img_name)), cv2.COLOR_BGR2RGB)
    
    # face alignment and segmentation：面部对齐和分割
    face_rgba = pre.process(img)
    if face_rgba is not None:
        # change background to white：背景置白
        face = face_rgba[:,:,:3].copy()
        mask = face_rgba[:,:,3].copy()[:,:,np.newaxis]/255.
        face_white_bg = (face*mask + (1-mask)*255).astype(np.uint8)
        #cv2.imwrite()将图像保存到指定文件
        cv2.imwrite(os.path.join(args.save_path, str(idx).zfill(4)+'.png'), cv2.cvtColor(face_white_bg, cv2.COLOR_RGB2BGR))
