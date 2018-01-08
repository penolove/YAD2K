import os
import time
import argparse
import pathlib
import requests
import arrow
import pygame
import pygame.camera
from pygame.locals import *

Debug = True
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

parser.set_defaults(save=True)

class WbcamCaputrePygame(object):
    def __init__(self, args):
        self.ip = args.ip
        self.BaseDir = args.BaseDir
        pygame.init()
        pygame.camera.init()

    def get_img_write_webcam(self):
        self.CAM = pygame.camera.Camera("/dev/video0",(640,480))
        self.CAM.start()
        image = self.CAM.get_image()
        self.CAM.stop()
        return image

    def post_yolo_path(self, img_path):
        print("posting path to yolo server")
        try:
            r = requests.post("http://%s/echo"%(self.ip), data='{"image_path": "'+img_path+'"}')
        except Exception as e:
            print("[post_yolo_path] post somewhat fails")
            print(e)
        print("post done")

    def write_image_sent_path(self):
        while True:
            # get time
            now = arrow.now(TZINFO)
            target_date = now.format('YYYY/MM/DD/HH')
            target_dir = os.path.join(args.BaseDir, target_date)
            pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
            image_name = now.format('mm_ss')
            img_path = '%s.jpg' % (os.path.join(target_dir, image_name))

            # get image
            image = self.get_img_write_webcam()
            # write to basedir/target_date/image_name
            if Debug: print("[write_image_sent_path] current writing to", img_path)
            pygame.image.save(image, img_path)

            # post target_date/image_name to yolo server
            img_path = '%s.jpg' % (os.path.join(target_date, image_name))
            self.post_yolo_path(img_path)
            time.sleep(5)


if __name__ == '__main__':
    args = parser.parse_args()
    WCPG = WbcamCaputrePygame(args)
    WCPG.write_image_sent_path()


