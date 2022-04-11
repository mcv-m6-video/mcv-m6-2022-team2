url=https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5x.pt
echo 'Downloading' $url$f && wget $url$f && mv *.pt data/weights/yolov5x.pt