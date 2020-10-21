from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense, Conv2D
from keras.layers import Input, MaxPooling2D
from keras.optimizers import Adam
from keras import Model
from PIL import ImageFile
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ----------------------------------------------------------------------------------------------------------------------

EPOCHS = 20

BATCH_SIZE = 32
NUM_CLASSES = 13
image_height = 224
image_width = 224
channels = 3
model_dir = "model/model_bak.h5"
train_dir = "./data/train/"
test_dir = "./data/test/"


# ----------------------------------------------------------------------------------------------------------------------

def get_datasets():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range = 90
    )

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(image_height, image_width),
                                                        color_mode="rgb",
                                                        batch_size=BATCH_SIZE,
                                                        class_mode="categorical")

    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range = 90
    )
    test_generator = test_datagen.flow_from_directory(test_dir,
                                                      target_size=(image_height, image_width),
                                                      color_mode="rgb",
                                                      batch_size=BATCH_SIZE,
                                                      class_mode="categorical"
                                                      )

    train_num = train_generator.samples
    test_num = test_generator.samples

    return train_generator, test_generator, train_num, test_num


# ----------------------------------------------------------------------------------------------------------------------

def VGG16(num_classes):
    image_input = Input(shape=(224, 224, 3))

    # ------------------------------------------------------------------------------------------------------------------
    # 第一个卷积部分
    # ------------------------------------------------------------------------------------------------------------------

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(image_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # ------------------------------------------------------------------------------------------------------------------
    # 第二个卷积部分
    # ------------------------------------------------------------------------------------------------------------------

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # ------------------------------------------------------------------------------------------------------------------
    # 第三个卷积部分
    # ------------------------------------------------------------------------------------------------------------------

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # ------------------------------------------------------------------------------------------------------------------
    # 第四个卷积部分
    # ------------------------------------------------------------------------------------------------------------------

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # ------------------------------------------------------------------------------------------------------------------
    # 第五个卷积部分
    # ------------------------------------------------------------------------------------------------------------------

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # ------------------------------------------------------------------------------------------------------------------
    # 全连接以及输出部分
    # ------------------------------------------------------------------------------------------------------------------

    x = Flatten(name='flatten')(x)
    x = Dense(256, activation='relu', name='fullc1')(x)
    x = Dense(256, activation='relu', name='fullc2')(x)
    x = Dense(num_classes, activation='softmax', name='fullc3')(x)
    model = Model(image_input, x, name='vgg16')

    return model


# ----------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    model = VGG16(NUM_CLASSES)
    model.load_weights('./model/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name=True)
    train_generator, test_generator, train_num, test_num = get_datasets()

    # ------------------------------------------------------------------------------------------------------------------

    trainable_layer = 19
    for i in range(trainable_layer):
        model.layers[i].trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])

    history=model.fit_generator(train_generator,
                        epochs=EPOCHS,
                        steps_per_epoch=train_num // BATCH_SIZE,
                        validation_data=test_generator,
                        validation_steps=test_num // BATCH_SIZE)

    model.save('./model/model_bak.h5')
    model.save_weights('./model/model_weights.h5')
    plt.subplot(2, 1, 1)
    epochs = len(history.history['loss'])
    plt.plot(range(epochs), history.history['loss'], label='loss')
    plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    epochs = len(history.history['acc'])
    plt.plot(range(epochs), history.history['acc'], label='acc')
    plt.plot(range(epochs), history.history['val_acc'], label='val_acc')
    plt.legend()
    plt.savefig("loss_acc.png")

# ----------------------------------------------------------------------------------------------------------------------
