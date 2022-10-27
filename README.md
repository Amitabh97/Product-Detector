
# Product Detector

Objective: Using Yolov7 I have tried to build an interactive product rating system where users can post picture/video of them with the custom product of my company and rate it either in like/dislike[thumbs up for 5 star, thumbs down for 1 star].



## Power of Yolov7

YOLO stands for “You Only Look Once”, it is a popular family of real-time object detection algorithms.
YOLOv7 is the latest official YOLO version created by the original authors of the YOLO architecture.
YOLOv7 is the fastest and most accurate real-time object detection model for computer vision tasks. The official YOLOv7 paper named “YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors” was released in July 2022 by Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark Liao.


![App Screenshot](https://viso.ai/wp-content/uploads/2022/08/computer-vision-in-aviation-viso-ai-1-1060x795.png)

Following is the official yolov7 github link : https://github.com/WongKinYiu/yolov7


## Creating Virtual Environment:

A virtual environment is created in anaconda prompt with following command

```bash
conda create -n detector python=3.9
conda activate detector

```
## Data Collection: 

Custom Images with my company products are collected by the following python code(in jupyter notebook) –

```bash
import uuid   #unique identifier
import os
import time
Images_path=os.path.join(‘data’,’images’)
labels = [‘1star’, ’5star’, ‘Book1’, ‘Book2’, ’Book3’, ‘Phone1’, ‘Phone2’, ‘Phone3’, ‘Speaker’, ‘Watch1’, ‘Watch2’]
number_img = 20
cap = cv2.VideoCapture(0)
#loop through labels
for label in labels:
    print(‘Collecting images for {}’.format(label))
    time.sleep(5)

     #loop through image range
     for img in range(number_img):
        print(‘Collecting images for {}, image number {}’.format(label, img))
        #webcam feed
        ret, frame = cap.read()
        #naming image
        imgname = os.path.join(Images_path, label+’.’+str(uuid.uuid1())+’.jpg’)
        #writes out image to file
        cv2.imwrite(imgname, frame)
        #render to the screen
        cv2.imshow(‘Image Collection’, frame)
        # 2 second delay between captures
        time.sleep(2)
        if cv2.waitkey(10) & 0xFF == ord(‘q’):
                   break
cap.release()
cv2.destroyAllWindows()
```

## Labelling Data: 

To label the custom products and ratings[thumbs up/down], we use “labelImg”. Following is the code to implement:
```bash
pip install labelImg
labelImg
```

## Model Selection: 
YOLOv7 is the fastest and most accurate real-time object detection model for computer vision tasks. There are 6 yolo models that can be used in version 7. Based on the limited computing power available in my system, I choose the base model i.e. YOLOv7. 

Download yolov7.pt from the official yolov7 github link: https://github.com/WongKinYiu/yolov7

## Model training: 
The following code will train the model – 
```bash
python train.py --workers 1 --device 0 --batch-size 16 --epochs 500 --img 640 640 --data data/custom_data.yaml --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/yolov7-custom.yaml --name yolov7-custom --weights yolov7.pt
```
The same is performed in google colab as well (for better performance). You can follow Detector.ipynb file for reference.

## Model Testing: 
Use the below code for testing the model on a new dataset
Dataset->video
```bash
python detect.py --weights best.pt --conf 0.5 --img-size 640 --source test1.mp4 --no-trace 
```


https://drive.google.com/file/d/1fVkb79QvTerq6XVlN13Vp_cwMQkDV9AY/view?usp=sharing

## Model Improvement

Our model performance can be improved by increasing the batch size and no of epochs which is limited now due to my system limitation(Ryzen 5 with Nvidia 1650Ti)/ google colab free gpu resources available.