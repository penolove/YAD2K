import os
import time
import cv2
import requests
import pathlib
import arrow
import tempfile
Debug = True


def get_img_write_webcam():
    # used in webcam,
    # for cv2 if release cam , will buffer image
    # with reintialize cam takes ~2s
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    if not ret:
        print("frame not get properly")
        return None
    cam.release()
    return frame

def post_yolo_path(img_path):
    print("posting path to yolo server")
    try:
        r = requests.post("http://localhost:5566/echo", data='{"image_path": "'+img_path+'"}')
    except Exception as e:
        print("[post_yolo_path] post somewhat fails")
        print(e)

def post_yolo_bytes(buf, output_name):
    print("posting buf to yolo server")
    try:
        headers = {'output_name': output_name}
        r = requests.post("http://localhost:5566/upload_images", data=buf, headers=headers)
    except Exception as e:
        print("[post_yolo_bytes] post somewhat fails")
        print(e)


def write_image_sent_path():
    while True:
        # get image
        frame = get_img_write_webcam()

        # get time
        now = arrow.now()
        target_dir = now.format('YYYY/MM/DD/HH')
        pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
        image_name = now.format('mm_ss')
        img_path = '%s.jpg' % (os.path.join(target_dir, image_name))
        if Debug: print("[write_image_sent_path] current writing to", img_path)
        cv2.imwrite(img_path, frame)
        post_yolo_path(img_path)
        time.sleep(5)


def sent_image_bytes():
    while True:
        # get image
        frame = get_img_write_webcam()
        if frame is not None:
            # get time
            now = arrow.now()
            target_dir = now.format('YYYY/MM/DD/HH')
            pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
            image_name = now.format('mm_ss')
            img_path = '%s.jpg' % (os.path.join(target_dir, image_name))
            retval, buffer = cv2.imencode('.jpg', frame)
            if Debug: print("[sent_image_bytes] current passing ", img_path)
            post_yolo_bytes(buffer.tobytes(), img_path)
        time.sleep(5)


if __name__ == '__main__':
    #write_image_sent_path()
    sent_image_bytes()


