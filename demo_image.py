# --------------------------------------------------------------------------------------------------------------------

from train import VGG16
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os
import numpy
from tensorflow.python.keras.backend import set_session
import tensorflow as tf
sess = tf.Session()
global graph, load_model
graph = tf.get_default_graph()
set_session(sess)
# --------------------------------------------------------------------------------------------------------------------

labels = ['apple', 'banana', 'pineapple', 'strawberry', 'watermelon', 'blueberry', 'durian', 'grape', 'Hami-melon', 'kiwi-fruit', 'litchi', 'orange', 'peach']
model = VGG16(13)
model.load_weights("./model/model_bak.h5")

# --------------------------------------------------------------------------------------------------------------------



# files = os.listdir('./test/')
# for file in files:
#     image_path = './img/' + file
#     print(image_path)
#     img = Image.open(image_path)
#     img = img.resize((224, 224))
#     img = np.array(img).reshape(-1, 224, 224, 3).astype('float32') / 255
#
#     prediction = model.predict(img)
#     final_prediction = [result.argmax() for result in prediction][0]
#     probability = np.max(prediction)
#
#     print(probability)
#     print(labels[final_prediction])
#     print('--------------')
#
#     image = cv2.imread(image_path)
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     # image = cv2ImgAddText(image, labels[final_prediction], 20, 30, (255, 23, 140), 20)
#     # cv2.putText(image, labels[final_prediction], (20, 30), font, 1.2, (255, 23, 140), 2)
#     cv2.imshow('', image)

    # cv2.waitKey(0)

# --------------------------------------------------------------------------------------------------------------------
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