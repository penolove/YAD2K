# python3 ./bottle_test_yolo.py model_data/yolo.h5 --ip localhost:5566 -o /MAS/pics/ -s 0.68
python3 ./bottle_test_yolo.py model_data/yolo.h5 --ip localhost:5566 -o /MAS/pics/ -s 0.68 --line_broadcast LineImageSender.json
