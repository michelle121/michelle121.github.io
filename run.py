import os
import cv2
import torch
from datetime import timedelta
from flask import Flask, render_template, request, jsonify

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from models import ResnetGenerator
import argparse
from utils import Preprocess

from flask import Flask, render_template, request,Response,jsonify
from flask_cors import CORS
import json
import base64

# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])

absolute_path = os.path.dirname(__file__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)


parser = argparse.ArgumentParser()
parser.add_argument('--photo_path', type=str, help='input photo path')
parser.add_argument('--save_path', type=str, help='cartoon save path')
args = parser.parse_args()

#os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

#os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

class Photo2Cartoon:
    def __init__(self):
        self.pre = Preprocess()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = ResnetGenerator(ngf=32, img_size=256, light=True).to(self.device)
        
        assert os.path.exists('./models/photo2cartoon_weights.pt'), "[Step1: load weights] Can not find 'photo2cartoon_weights.pt' in folder 'models!!!'"
        params = torch.load('./models/photo2cartoon_weights.pt', map_location=self.device)
        self.net.load_state_dict(params['genA2B'])
        print('[Step1: load weights] success!')

        
    def inference(self, img):
        face_rgba = self.pre.process(img)
        if face_rgba is None:
            print('[Step2: face detect] can not detect face!!!')
            return None
        
        print('[Step2: face detect] success!')

        face_rgba = cv2.resize(face_rgba, (256, 256), interpolation=cv2.INTER_AREA)
       
        
        face = face_rgba[:, :, :3].copy()
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


@app.route('/', methods=['POST', 'GET'])  # 添加路由
def upload():
    if request.method == 'POST':
        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        current_path = os.path.dirname(__file__)  # 当前文件所在路径

        upload_path = os.path.join(current_path, 'static/image_input', 'test.jpg')
        f.save(upload_path)
        
        img = cv2.cvtColor(cv2.imread(upload_path), cv2.COLOR_BGR2RGB)
        cartoon = c2p.inference(img)
        if cartoon is not None:
            cv2.imwrite('static/image_output/out.png', cartoon)
        return render_template('recognition.html')

    return render_template('index.html')


CORS(app, supports_credentials=True)
@app.route('/1', methods=['POST', 'GET']) 
def receive_image():
 
    if request.method == "POST":
        data = request.data.decode('utf-8')
        json_data = json.loads(data)
        str_image = json_data.get("imgData")
        img = base64.b64decode(str_image)
        img_np = np.fromstring(img, dtype='uint8')
        new_img_np = cv2.imdecode(img_np, 1)
        cv2.imwrite('static/image_input/test6.jpg', new_img_np)


        img = cv2.cvtColor(new_img_np, cv2.COLOR_BGR2RGB)
        cartoon = c2p.inference(img)
        if cartoon is not None:
            cv2.imwrite('.static/image_output/out.png',cartoon)
            print('data:{}'.format('success'))
        return render_template('recognition.html')
        
    #return render_template('recognition.html')
    #return Response('upload')
    return render_template('camera.html')


if __name__ == '__main__':
    c2p = Photo2Cartoon()
    app.run(debug=False)
    print('Cartoon portrait has been saved successfully!')




