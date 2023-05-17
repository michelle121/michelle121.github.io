from flask import Flask, render_template, request,Response,jsonify
from flask_cors import CORS
import json
import cv2
import numpy as np
import base64
app = Flask(__name__)
CORS(app, supports_credentials=True)
 
'''
@app.route('/')
def hello_world():
 
    return Response('hello_world')
'''
'''
@app.route('/video_sample/')
def video_sample():
 
    return render_template('camera.html')
 '''
 
@app.route('/', methods=['POST', 'GET']) 
def receive_image():
 
    if request.method == "POST":
        data = request.data.decode('utf-8')
        json_data = json.loads(data)
        str_image = json_data.get("imgData")
        img = base64.b64decode(str_image)
        img_np = np.fromstring(img, dtype='uint8')
        new_img_np = cv2.imdecode(img_np, 1)
        cv2.imwrite('.static/image_saved_rev_image.jpg',new_img_np)
        print('data:{}'.format('success'))
 
    #return Response('upload')
    return render_template('camera.html')
 
if __name__ == '__main__':
 
    #app.run(debug=True,host='0.0.0.0',port=5000)
    app.run(debug=False)