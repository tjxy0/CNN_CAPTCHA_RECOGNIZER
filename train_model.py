# -*- coding: utf-8 -*-
import json

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import os
from cnnlib.network import CNN  # 假设这是迁移后的CNN类
from PIL import Image

class TrainError(Exception):
    pass


class TrainModel(CNN):
    def __init__(self, train_img_path, verify_img_path, char_set, model_save_dir, cycle_stop, acc_stop, cycle_save,
                 image_suffix, train_batch_size, test_batch_size, verify=False):
        # 训练相关参数
        self.cycle_stop = cycle_stop
        self.acc_stop = acc_stop
        self.cycle_save = cycle_save
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.image_suffix = image_suffix
        self.model_save_dir = model_save_dir
        char_set = [str(i) for i in char_set]

        # 处理训练集
        self.train_img_path = train_img_path
        self.train_images_list = os.listdir(train_img_path)
        if verify:
            self.confirm_image_suffix()
        random.seed(int(time.time()))
        random.shuffle(self.train_images_list)

        # 处理验证集
        self.verify_img_path = verify_img_path
        self.verify_images_list = os.listdir(verify_img_path)

        # 获取图片基本信息
        label, captcha_array = self.gen_captcha_text_image(train_img_path, self.train_images_list[0])
        if len(captcha_array.shape) == 3:
            image_height, image_width, _ = captcha_array.shape
        else:
            image_height, image_width = captcha_array.shape

        # 初始化父类
        super().__init__(image_height, image_width, len(label), char_set, model_save_dir)

        # 打印信息
        print(f"\n{'=' * 30} 训练参数 {'=' * 30}")
        print(f"图片尺寸: {image_height}x{image_width}")
        print(f"验证码长度: {self.max_captcha}")
        print(f"字符类别数: {self.char_set_len}")
        print(f"训练集路径: {train_img_path}")
        print(f"验证集路径: {verify_img_path}\n")

    @staticmethod
    def gen_captcha_text_image(img_path, img_name):
        """获取图片和标签"""
        label = img_name.split("_")[0]
        img_file = os.path.join(img_path, img_name)
        captcha_image = Image.open(img_file)
        return label, np.array(captcha_image)

    def get_batch(self, size=128):
        """生成训练批次数据"""
        batch_x = np.zeros([size, self.image_height * self.image_width])
        batch_y = np.zeros([size, self.max_captcha * self.char_set_len])

        selected_images = random.sample(self.train_images_list, size)
        for i, img_name in enumerate(selected_images):
            label, image_array = self.gen_captcha_text_image(self.train_img_path, img_name)
            image_array = self.convert2gray(image_array)
            batch_x[i, :] = image_array.flatten() / 255
            batch_y[i, :] = self.text2vec(label)
        return batch_x, batch_y

    def get_verify_batch(self, size=100):
        """生成验证批次数据"""
        batch_x = np.zeros([size, self.image_height * self.image_width])
        batch_y = np.zeros([size, self.max_captcha * self.char_set_len])

        selected_images = random.sample(self.verify_images_list, size)
        for i, img_name in enumerate(selected_images):
            label, image_array = self.gen_captcha_text_image(self.verify_img_path, img_name)
            image_array = self.convert2gray(image_array)
            batch_x[i, :] = image_array.flatten() / 255
            batch_y[i, :] = self.text2vec(label)
        return batch_x, batch_y

    def confirm_image_suffix(self):
        """校验图片格式"""
        print("开始校验图片格式...")
        for img_name in self.train_images_list:
            if not img_name.endswith(self.image_suffix):
                raise TrainError(f'文件 [{img_name}] 格式不匹配，要求格式 [{self.image_suffix}]')
        print("所有图片格式校验通过")

    def train_cnn(self):
        """训练模型"""
        # 定义优化器和损失函数
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # 创建指标跟踪器
        train_char_acc = tf.keras.metrics.Mean()
        train_image_acc = tf.keras.metrics.Mean()
        val_char_acc = tf.keras.metrics.Mean()
        val_image_acc = tf.keras.metrics.Mean()

        # 训练循环
        best_acc = 0.0
        for epoch in range(1, self.cycle_stop + 1):
            # 训练步骤
            batch_x, batch_y = self.get_batch(size=self.train_batch_size)
            with tf.GradientTape() as tape:
                logits = self(batch_x, training=True)
                loss = loss_fn(batch_y, logits)

            grads = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.trainable_variables))

            # 计算训练准确率
            acc_char, acc_image = self.calculate_accuracy(logits, batch_y)
            train_char_acc(acc_char)
            train_image_acc(acc_image)

            # 验证步骤
            if epoch % 10 == 0:
                val_x, val_y = self.get_verify_batch(size=self.test_batch_size)
                val_logits = self(val_x, training=False)
                val_loss = loss_fn(val_y, val_logits)

                val_acc_char, val_acc_image = self.calculate_accuracy(val_logits, val_y)
                val_char_acc(val_acc_char)
                val_image_acc(val_acc_image)

                # 打印训练信息
                print(f"\nEpoch {epoch}/{self.cycle_stop}")
                print(
                    f"[Train] loss: {loss:.4f} - char_acc: {train_char_acc.result():.4f} - image_acc: {train_image_acc.result():.4f}")
                print(
                    f"[Val]   loss: {val_loss:.4f} - char_acc: {val_char_acc.result():.4f} - image_acc: {val_image_acc.result():.4f}")

                # 重置指标
                train_char_acc.reset_states()
                train_image_acc.reset_states()
                val_char_acc.reset_states()
                val_image_acc.reset_states()

                # 保存最佳模型
                if val_acc_image > best_acc:
                    best_acc = val_acc_image
                    self.save_weights(self.model_save_dir)
                    print(f"验证准确率提升至 {best_acc:.4f}, 模型已保存")

                # 提前停止
                if val_acc_image >= self.acc_stop:
                    print(f"\n验证准确率达到 {self.acc_stop}, 停止训练")
                    return

            # 定期保存
            if epoch % self.cycle_save == 0:
                self.save_weights(f"{self.model_save_dir}_epoch_{epoch}")
                print(f"周期保存模型：{self.model_save_dir}_epoch_{epoch}")

    def calculate_accuracy(self, logits, labels):
        """计算字符级和图片级准确率"""
        # 转换形状 [batch, max_captcha*char_set_len] => [batch, max_captcha, char_set_len]
        predict = tf.reshape(logits, [-1, self.max_captcha, self.char_set_len])
        label = tf.reshape(labels, [-1, self.max_captcha, self.char_set_len])

        # 计算字符级准确率
        char_acc = tf.reduce_mean(tf.cast(tf.argmax(predict, -1) == tf.argmax(label, -1), tf.float32))

        # 计算图片级准确率（需要所有字符正确）
        image_acc = tf.reduce_mean(tf.cast(tf.reduce_all(
            tf.argmax(predict, -1) == tf.argmax(label, -1), axis=1), tf.float32))

        return char_acc, image_acc

    def recognize_captcha(self):
        """验证码识别示例"""
        # 随机选择测试图片
        img_name = random.choice(self.train_images_list)
        label, captcha_array = self.gen_captcha_text_image(self.train_img_path, img_name)

        # 预处理图片
        image = self.convert2gray(captcha_array)
        image_input = image.flatten() / 255
        image_input = np.expand_dims(image_input, axis=0)  # 添加batch维度

        # 进行预测
        logits = self.predict(image_input)
        predict = tf.argmax(tf.reshape(logits, [-1, self.max_captcha, self.char_set_len]), -1)
        predict_text = ''.join([self.char_set[i] for i in predict.numpy()[0]])

        # 可视化结果
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(captcha_array)
        plt.title(f"Original: {label}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(image, cmap='gray')
        plt.title(f"Prediction: {predict_text}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def main():
    with open("conf/sample_config.json", "r") as f:
        sample_conf = json.load(f)

    # 配置参数
    config = {
        "train_img_path": sample_conf["train_image_dir"],
        "verify_img_path": sample_conf["test_image_dir"],
        "model_save_dir": sample_conf["model_save_dir"],
        "cycle_stop": sample_conf["cycle_stop"],
        "acc_stop": sample_conf["acc_stop"],
        "cycle_save": sample_conf["cycle_save"],
        "image_suffix": sample_conf['image_suffix'],
        "train_batch_size": sample_conf['train_batch_size'],
        "test_batch_size": sample_conf['test_batch_size'],
        "char_set": json.load(open("tools/labels.json")) if sample_conf['use_labels_json_file']
        else sample_conf["char_set"]
    }

    # GPU设置
    if not sample_conf["enable_gpu"]:
        tf.config.set_visible_devices([], 'GPU')

    # 初始化训练器
    trainer = TrainModel(**config)

    # 加载已有模型（如果存在）
    if os.path.exists(config["model_save_dir"]):
        try:
            trainer.load_weights(config["model_save_dir"])
            print("成功加载已有模型")
        except:
            print("未找到有效模型，开始新训练")

    # 开始训练
    trainer.train_cnn()

    # 测试识别效果
    trainer.recognize_captcha()


if __name__ == '__main__':
    main()