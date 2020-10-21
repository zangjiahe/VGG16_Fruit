import subprocess

from train import VGG16
from PIL import Image
import numpy as np
import cv2
from tensorflow.python.keras.backend import set_session
import tensorflow as tf
import _thread
sess = tf.Session()
global graph, load_model
graph = tf.get_default_graph()
set_session(sess)

labels = ['apple', 'banana', 'pineapple', 'strawberry', 'watermelon', 'blueberry', 'durian', 'grape', 'Hami-melon', 'kiwi-fruit', 'litchi', 'orange', 'peach']
model = VGG16(13)
model.load_weights("./model/model_bak.h5")


rtmp = 'rtmp://46132.livepush.myqcloud.com/live/video?txSecret=f1bb18a3b5176fc3593cc7f7065cdfce&txTime=5FC5F3FC'

# 读取视频并获取属性
cap = cv2.VideoCapture(0)#"rtmp://live.bcaqfy.xin/live/zangjiahe"
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

def predictImg(path):
    image_path = path
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = np.array(img).reshape(-1, 224, 224, 3).astype('float32') / 255
    with graph.as_default():
        set_session(sess)
        prediction = model.predict(img)
    final_prediction = [result.argmax() for result in prediction][0]
    probability = np.max(prediction)
    probability=float(probability)*100
    return labels[final_prediction]+"|"+str(round(probability, 2))


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
                prediction=model.predict(frame)
                result = labels[int(np.argmax(prediction))]
            font = cv2.FONT_HERSHEY_SIMPLEX
            w=int(int(witdh) / 2)
            h=int(int(height)/2)
            probability = round(float(np.max(prediction))*100,2)
            cv2.putText(ori, "result:"+result, (30,60), font, 1.2, (255, 23, 140), 2)
            cv2.putText(ori, "confirm:" + str(probability)+"%", (30, 100), font, 1.2, (255, 23, 140), 2)
            cv2.putText(ori, "fps:" + str(cap.get(cv2.CAP_PROP_FPS)), (30, 140), font, 1.2, (255, 23, 140), 2)
            cv2.imshow('', ori)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        pipe.stdin.write(ori.tostring())

    cap.release()
    pipe.terminate()


