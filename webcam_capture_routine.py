import os
import time
import cv2
import requests
import pathlib
import arrow
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

dir_queue = []
def post_yolo(prefix,min_path):
    dir_queue.insert(0,min_path)
    if len(dir_queue)>1:
        try:
            dir_path = dir_queue.pop()
            r = requests.post("http://localhost:5566/home", data='{"dir_path": "'+prefix+dir_path+'"}')
        except Exception as e:
            print("====[file monitor] post somewhat fails===")
            print(e)

if __name__ == '__main__':
    while True:
        # get image
        frame = get_img_write_webcam()

        # get time
        now = arrow.now()
        target_dir = now.format('YYYY/MM/DD/HH')
        pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
        image_name = now.format('mm_ss')
        img_path = '%s.jpg' % (os.path.join(target_dir, image_name))
        if Debug: print("current writing to", img_path)
        cv2.imwrite(img_path, frame)
        time.sleep(5)





