import cv2

import subprocess

from train import VGG16
import numpy as np
from tensorflow.python.keras.backend import set_session
import tensorflow as tf
sess = tf.Session()
global graph, load_model
graph = tf.get_default_graph()
set_session(sess)

labels = ['apple', 'banana', 'pineapple', 'strawberry', 'watermelon', 'blueberry', 'durian', 'grape', 'Hami-melon', 'kiwi-fruit', 'litchi', 'orange', 'peach'] #['苹果', '香蕉', '菠萝', '草莓', '西瓜', '蓝莓', '榴莲', '葡萄', '哈密瓜', '猕猴桃', '荔枝', '橘子', '桃子']
model = VGG16(13)
model.load_weights("./model/model_bak.h5")


rtmp = 'rtmp://46132.livepush.myqcloud.com/live/video?txSecret=f1bb18a3b5176fc3593cc7f7065cdfce&txTime=5FC5F3FC'

# 读取视频并获取属性
cap = cv2.VideoCapture("rtmp://live.bcaqfy.xin/live/zangjiahe")
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
sizeStr = str(size[0]) + '*' + str(size[1])
witdh=str(size[0])
height=str(size[1])
command = [
    'ffmpeg',
           '-y', '-an',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-pix_fmt', 'bgr24',
           '-s', sizeStr,
           '-r', '25',
           '-i', '-',
           '-c:v', 'libx264',
           '-pix_fmt', 'yuv420p',
           '-preset', 'ultrafast',
           '-f', 'flv',
           rtmp]

pipe = subprocess.Popen(command
                        , shell=False
                        , stdin=subprocess.PIPE
                        )
def star():
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            '''
            对frame进行识别处理
            '''
            ori = frame
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame / 255
            frame = np.expand_dims(frame, axis=0)
            with graph.as_default():
                set_session(sess)
                result = labels[int(np.argmax(model.predict(frame)))]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(ori, result, (200,300), font, 1.2, (255, 23, 140), 2)
            cv2.imshow('', ori)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        pipe.stdin.write(ori.tostring())

    cap.release()
    pipe.terminate()
