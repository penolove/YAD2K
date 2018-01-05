#! /usr/bin/env python
"""Run a YOLO_v2 style detection model on test images."""
import io
import os
import time
import json
import imghdr
import random
import pathlib
import sqlite3
import argparse
import colorsys
import arrow

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
from bottle import route, run, request

from yad2k.models.keras_yolo import yolo_eval, yolo_head

parser = argparse.ArgumentParser(
    description='Run a YOLO_v2 style detection model on test images..')
parser.add_argument(
    'model_path',
    help='path to h5 model file containing body'
    'of a YOLO_v2 model')
parser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to yolo_anchors.txt',
    default='model_data/yolo_anchors.txt')
parser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to coco_classes.txt',
    default='model_data/coco_classes.txt')
parser.add_argument(
    '-t',
    '--test_path',
    help='path to directory of test images, defaults to images/',
    default='images')
parser.add_argument(
    '-o',
    '--output_path',
    help='path to output test images, defaults to images/out',
    default='out')
parser.add_argument(
    '-s',
    '--score_threshold',
    type=float,
    help='threshold for bounding box scores, default .3',
    default=.3)
parser.add_argument(
    '-iou',
    '--iou_threshold',
    type=float,
    help='threshold for non max suppression IOU, default .5',
    default=.5)
parser.add_argument(
    '-db_path',
    '--db_path',
    help='datapath of db',
    default='default_yolo.sqlite3')

def _main(args):
    yolo_model = YoloModel(args)
    # run test images
    #yolo_model.detect_test_folder()

    @route('/echo', method='POST')
    def echo():
        data = request.body.read()
        body = json.loads(data.decode())
        im_path = body['image_path']
        arrive_timestamp = arrow.now().datetime
        yolo_model.insert_image_info(im_path, arrive_timestamp)  # insert into image_info
        yolo_model.image_detection(im_path) # detect, save into annoation

    @route('/folder_detection', method='POST')
    def folder_detection():
        data = request.body.read()
        body = json.loads(data.decode())
        dir_path = body['dir_path']
        arrive_timestamp = arrow.now().datetime
        yolo_model.detect_images_in_folder(dir_path, arrive_timestamp)

    @route('/upload_images', method='POST')
    def upload_image():
        """make sure you add output_name in post header"""
        data = request.body.read()
        file_name = request.headers.get('output_name', 'temp.jpg')
        im_path = io.BytesIO(bytearray(data))
        yolo_model.image_detection(im_path, post_image_name=file_name)

    run(host='localhost', port=5566, debug=True)

class YoloModel(object):
    def __init__(self, args):
        model_path = os.path.expanduser(args.model_path)
        assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
        anchors_path = os.path.expanduser(args.anchors_path)
        classes_path = os.path.expanduser(args.classes_path)
        self.test_path = os.path.expanduser(args.test_path)
        self.output_path = os.path.expanduser(args.output_path)


        if not os.path.exists(self.output_path):
            print('Creating output path {}'.format(self.output_path))
            os.mkdir(self.output_path)

        self.sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

        with open(classes_path) as f:
            class_names = f.readlines()
        self.class_names = [c.strip() for c in class_names]

        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)

        self.yolo_model = load_model(model_path)

        # Verify model, anchors, and classes are compatible
        num_classes = len(class_names)
        num_anchors = len(anchors)
        # TODO: Assumes dim ordering is channel last
        model_output_channels = self.yolo_model.layers[-1].output_shape[-1]
        assert model_output_channels == num_anchors * (num_classes + 5), \
            'Mismatch between model and given anchor and class sizes. ' \
            'Specify matching anchors and classes with --anchors_path and ' \
            '--classes_path flags.'
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Check if model is fully convolutional, assuming channel last order.
        self.model_image_size = self.yolo_model.layers[0].input_shape[1:3]
        self.is_fixed_size = self.model_image_size != (None, None)

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(class_names), 1., 1.)
                      for x in range(len(class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        # TODO: Wrap these backend operations with Keras layers.
        yolo_outputs = yolo_head(self.yolo_model.output, anchors, len(class_names))
        self.input_image_shape = K.placeholder(shape=(2, ))
        self.boxes, self.scores, self.classes = yolo_eval(
            yolo_outputs,
            self.input_image_shape,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold)

        # connet/create db, add table(if not exist)
        self.conn = sqlite3.connect(args.db_path)

        print('connet/create image_info table')
        self.conn.execute('''CREATE TABLE IF NOT EXISTS image_info(
                          image_path TEXT PRIMARY KEY,
                          timestamp TIMESTAMP
                          )''')
        print('connet/create image_annotation table')
        self.conn.execute('''CREATE TABLE IF NOT EXISTS image_annotation(
                          image_path TEXT,
                          x1 INTEGER,
                          y1 INTEGER,
                          x2 INTEGER,
                          y2 INTEGER,
                          label TEXT,
                          FOREIGN KEY(image_path) REFERENCES image_info(image_path)
                          )''')
        self.conn.commit()

    def detect_test_folder(self):
        #self.detect_images_in_folder(self.test_path)
        arrive_timestamp = arrow.now().datetime
        self.detect_images_in_folder(self.test_path, arrive_timestamp)

    def detect_images_in_folder(self, folder_path, arrive_timestamp):
        for image_file in os.listdir(folder_path):
            try:
                image_type = imghdr.what(os.path.join(folder_path, image_file))
                if not image_type:
                    continue
            except IsADirectoryError:
                continue
            test_image_path = os.path.join(folder_path, image_file)
            self.insert_image_info(test_image_path, arrive_timestamp)
            self.image_detection(test_image_path)

    def insert_image_info(self, image_path, arrive_timestamp):
        self.conn.execute('''INSERT INTO image_info(image_path, timestamp)
                             VALUES(?,?)''', (image_path, arrive_timestamp))

    def image_detection(self, test_image_path, post_image_name=None):
        """image_detection
            give image_path return result of detections

        Parameters
        ----------
        test_image_path: str
            image_path.
            (can pass io.bytesio object, take a look upload_image in the main)

        post_image_name: str
            filename to save, this is used for post io.bytesio , both saving upload file and output

        Returns
        ---------
        detected_result_list: List[(str, (int, int), (int, int))]
            detected result List[(class, (x1, y1), (x2, y2))]
        """

        image = Image.open(test_image_path)
        from_post = not isinstance(test_image_path, str)
        if from_post:
            test_image_path = post_image_name
            directory = os.path.dirname(test_image_path)
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
            image.save(test_image_path, quality=90)
        start_time = time.time()
        if self.is_fixed_size:  # TODO: When resizing we can use minibatch input.
            resized_image = image.resize(
                tuple(reversed(self.model_image_size)), Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
        else:
            # Due to skip connection + max pooling in YOLO_v2, inputs must have
            # width and height as multiples of 32.
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            resized_image = image.resize(new_image_size, Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
            print(image_data.shape)

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        print('Found {} boxes for {}'.format(len(out_boxes), test_image_path))

        font = ImageFont.truetype(
            font='font/FiraMono-Medium.otf',
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        detected_result_list = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)

            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = int(max(0, np.floor(top + 0.5).astype('int32')))
            left = int(max(0, np.floor(left + 0.5).astype('int32')))
            bottom = int(min(image.size[1], np.floor(bottom + 0.5).astype('int32')))
            right = int(min(image.size[0], np.floor(right + 0.5).astype('int32')))
            print(label, (left, top), (right, bottom))
            detected_result_list.append((predicted_class, (left, top), (right, bottom)))
            # save prediction result if not from post method

            try:
                self.conn.execute('''INSERT INTO image_annotation(image_path, x1, y1, x2, y2
                                    ,label) VALUES (?,?,?,?,?,?)''',
                                    (test_image_path, left, top, right, bottom, predicted_class))
                self.conn.commit()
            except sqlite3.IntegrityError:
                pass

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw


            image_outpath = os.path.join(self.output_path, test_image_path)

            directory = os.path.dirname(image_outpath)
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
            image.save(image_outpath, quality=90)
            print("--- %s seconds ---" % (time.time() - start_time))

        return detected_result_list

    def __del__(self):
        self.sess.close()
        self.conn.close()

if __name__ == '__main__':
    _main(parser.parse_args())
