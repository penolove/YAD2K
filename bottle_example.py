import json
from bottle import route, run, request
import numpy as np
import cv2

@route('/home', method='POST')
def home():
    print("hello")

@route('/echo', method='POST')
def echo():
    data = request.body.read()
    print(data)
    body = json.loads(data.decode())
    im_path = body['dir_path']
    print("target path", im_path)


@route('/upload_images', method='POST')
def upload_image():
    data = request.body.read()
    #print(data)
    image = np.asarray(bytearray(data), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    cv2.imwrite('dog.jpg',image)
    print(image.shape)


run(host='localhost', port=5566, debug=True)