import tensorflow as tf
from tensorflow.keras import layers, Model, initializers
import numpy as np
import os
from PIL import Image
import random


class CNN(Model):
    def __init__(self, image_height, image_width, max_captcha, char_set, model_save_dir):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.max_captcha = max_captcha
        self.char_set = char_set
        self.char_set_len = len(char_set)
        self.model_save_dir = model_save_dir

        # 卷积层
        self.conv1 = layers.Conv2D(32, 3, padding='same', activation='relu',
                                   kernel_initializer=initializers.GlorotNormal())
        self.pool1 = layers.MaxPooling2D(pool_size=2, strides=2)
        self.drop1 = layers.Dropout(0.5)

        self.conv2 = layers.Conv2D(64, 3, padding='same', activation='relu',
                                   kernel_initializer=initializers.GlorotNormal())
        self.pool2 = layers.MaxPooling2D(pool_size=2, strides=2)
        self.drop2 = layers.Dropout(0.5)

        self.conv3 = layers.Conv2D(128, 3, padding='same', activation='relu',
                                   kernel_initializer=initializers.GlorotNormal())
        self.pool3 = layers.MaxPooling2D(pool_size=2, strides=2)
        self.drop3 = layers.Dropout(0.5)

        # 全连接层
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(1024, activation='relu',
                                   kernel_initializer=initializers.GlorotNormal())
        self.drop4 = layers.Dropout(0.5)
        self.dense2 = layers.Dense(max_captcha * self.char_set_len,
                                   kernel_initializer=initializers.GlorotNormal())

    def call(self, inputs, training=False):
        x = tf.reshape(inputs, [-1, self.image_height, self.image_width, 1])
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.drop1(x, training=training)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.drop2(x, training=training)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.drop3(x, training=training)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.drop4(x, training=training)
        return self.dense2(x)

    @staticmethod
    def convert2gray(img):
        if len(img.shape) > 2:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            return gray
        return img

    def text2vec(self, text):
        text_len = len(text)
        if text_len > self.max_captcha:
            raise ValueError(f'验证码最长{self.max_captcha}个字符')

        vector = np.zeros(self.max_captcha * self.char_set_len)
        for i, ch in enumerate(text):
            idx = i * self.char_set_len + self.char_set.index(ch)
            vector[idx] = 1
        return vector


# 使用示例
if __name__ == "__main__":
    # 参数配置
    image_height = 60
    image_width = 160
    max_captcha = 4
    char_set = "0123456789abcdefghijklmnopqrstuvwxyz"
    model_save_dir = "./model"

    # 初始化模型
    model = CNN(image_height, image_width, max_captcha, char_set, model_save_dir)

    # 定义优化器和损失函数
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # 编译模型（用于Keras训练接口）
    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=['accuracy'])

    # 打印模型结构
    model.build(input_shape=(None, image_height * image_width))
    print(model.summary())