import os
import time
import cv2
import argparse
import requests
import pathlib
import arrow
import tempfile
DEBUG = True
TZINFO = '+08:00'

parser = argparse.ArgumentParser(
    description='post image to bottle yolo server')

parser.add_argument(
    '-o',
    '--BaseDir',
    help='used for write_image_sent_path',
    default='')
parser.add_argument(
    '--ip',
    help='ip address used to post image',
    default='localhost:5566')

parser.add_argument('--nosave', dest='save', action='store_false')
parser.set_defaults(save=True)


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

def post_yolo_path(img_path, args):
    print("posting path to yolo server")
    try:
        headers = {'image_path': img_path,
                   'tzinfo': TZINFO}
        requests.post("http://%s/echo"%(args.ip), headers=headers)

    except Exception as e:
        print(e)
        print("[post_yolo_path] post somewhat fails")

    print("post done")

def post_yolo_bytes(buf, img_path, args):
    print("posting buf to yolo server")
    try:
        headers = {'img_path': img_path,
                   'tzinfo': TZINFO}
        requests.post("http://%s/upload_images"%(args.ip), data=buf, headers=headers)

    except Exception as e:
        print(e)
        print("[post_yolo_bytes] post somewhat fails")

    print("post done")

def write_image_sent_path(args):
    while True:
        # get image
        frame = get_img_write_webcam()

        # get time
        now = arrow.now(TZINFO)
        target_date = now.format('YYYY/MM/DD/HH')
        target_dir = os.path.join(args.BaseDir, target_date)
        pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
        image_name = now.format('mm_ss')
        img_path = '%s.jpg' % (os.path.join(target_dir, image_name))
        if DEBUG: print("[write_image_sent_path] current writing to", img_path)
        # write to basedir/target_date/image_name
        cv2.imwrite(img_path, frame)

        # post target_date/image_name to yolo server
        img_path = '%s.jpg' % (os.path.join(target_date, image_name))
        post_yolo_path(img_path, args)
        time.sleep(5)

def sent_image_bytes():
    while True:
        # get image
        frame = get_img_write_webcam()
        if frame is not None:
            # get time
            now = arrow.now(TZINFO)
            target_dir = now.format('YYYY/MM/DD/HH')
            image_name = now.format('mm_ss')
            img_path = '%s.jpg' % (os.path.join(target_dir, image_name))
            _, buffer = cv2.imencode('.jpg', frame)
            if DEBUG: print("[sent_image_bytes] current passing ", img_path)
            post_yolo_bytes(buffer.tobytes(), img_path, args)
        time.sleep(15)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.save:
        write_image_sent_path(args)
    else:
        sent_image_bytes()
