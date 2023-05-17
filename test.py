import os #对文件及文件夹进行操作
import torch
#减少需求防止报错，等程序运行完，在跑下一个程序
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import cv2
from models import ResnetGenerator
#导入模块用于于创建解析对象，在多个文件或者不同语言协同的项目中，python脚本经常需要从命令行直接读取参数。
import argparse
#utils存放自己写好的自定义函数的包
from utils import Preprocess


parser = argparse.ArgumentParser()
#向该对象中添加你要关注的命令行参数和选项
parser.add_argument('--photo_path', type=str, help='input photo path')
parser.add_argument('--save_path', type=str, help='cartoon save path')
#进行解析
args = parser.parse_args()

#创建多层目录；exist_ok：是否在目录存在时触发异常（Ture：不会）；
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

class Photo2Cartoon:
    def __init__(self):#用于初始化类且自动执行；self：实例本身不是类，不可省略
        self.pre = Preprocess()
        #选择将数据存储和转换分配到的设备的对象
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #残差网络生成器
        #ngf：第一层生成器中提取出的输出特征的数量
        self.net = ResnetGenerator(ngf=32, img_size=256, light=True).to(self.device)
        
        #判断文件\文件路径是否存在
        assert os.path.exists('./models/photo2cartoon_weights.pt'), "[Step1: load weights] Can not find 'photo2cartoon_weights.pt' in folder 'models!!!'"
        #加载模型；map_location:指定如何重新映射存储位置的函数
        params = torch.load('./models/photo2cartoon_weights.pt', map_location=self.device)
        #将预训练的参数权重加载到新的模型之中
        #params指定采用数目可变的参数的方法参数
        self.net.load_state_dict(params['genA2B'])
        print('[Step1: load weights] success!')

        #推理
    def inference(self, img):
        # face alignment and segmentation面部对齐和分割
        face_rgba = self.pre.process(img)
        if face_rgba is None:
            print('[Step2: face detect] can not detect face!!!')
            return None
        
        print('[Step2: face detect] success!')

        #将原始图像调整成指定大小；输入图像；输出图像大小；interpolation:插入方式：使用像素区域关系进行重采样
        #rgba:红，绿，蓝，透明度
        face_rgba = cv2.resize(face_rgba, (256, 256), interpolation=cv2.INTER_AREA)
       
        
        face = face_rgba[:, :, :3].copy()
        #掩膜：裁剪图像中任意形状的区域
        #np.newaxis:在这一位置增加一个一维
        mask = face_rgba[:, :, 3][:, :, np.newaxis].copy() / 255.
        face = (face*mask + (1-mask)*255) / 127.5 - 1
        #转置
        face = np.transpose(face[np.newaxis, :, :, :], (0, 3, 1, 2)).astype(np.float32)
        face = torch.from_numpy(face).to(self.device)

        # inference
        with torch.no_grad():
            cartoon = self.net(face)[0][0]

        # post-process
        cartoon = np.transpose(cartoon.cpu().numpy(), (1, 2, 0))
        cartoon = (cartoon + 1) * 127.5
        cartoon = (cartoon * mask + 255 * (1 - mask)).astype(np.uint8)
        cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
        print('[Step3: photo to cartoon] success!')
        return cartoon


if __name__ == '__main__':
    img = cv2.cvtColor(cv2.imread(args.photo_path), cv2.COLOR_BGR2RGB)
    c2p = Photo2Cartoon()
    print('[[[[[[[[]]]]]]')
    cartoon = c2p.inference(img)
    print('[[[[[[[[]]]]]]')
    if cartoon is not None:
        cv2.imwrite(args.save_path, cartoon)
        print('Cartoon portrait has been saved successfully!')
