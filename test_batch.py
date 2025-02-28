# -*- coding: utf-8 -*-
import json
import tensorflow as tf
import numpy as np
import time
import random
import os
from PIL import Image
from cnnlib.network import CNN  # 假设这是已迁移到TF2的CNN类


class TestError(Exception):
    pass


class TestBatch(CNN):
    def __init__(self, img_path, char_set, model_save_dir, total):
        # 初始化测试参数
        self.model_save_dir = model_save_dir
        self.img_path = img_path
        self.img_list = os.listdir(img_path)
        random.seed(int(time.time()))
        random.shuffle(self.img_list)
        self.total = total

        # 获取样本图片基本信息
        label, captcha_array = self.gen_captcha_text_image()
        if len(captcha_array.shape) == 3:
            image_height, image_width, _ = captcha_array.shape
        else:
            image_height, image_width = captcha_array.shape

        # 初始化父类
        super().__init__(image_height, image_width, len(label), char_set, model_save_dir)

        # 加载预训练权重
        try:
            self.load_weights(self.model_save_dir)
            print("成功加载预训练模型")
        except:
            raise TestError("无法加载模型权重，请检查路径: {}".format(self.model_save_dir))

        # 打印配置信息
        print("\n" + "=" * 30 + " 测试配置 " + "=" * 30)
        print(f"测试集路径: {img_path}")
        print(f"图片尺寸: {image_height}x{image_width}")
        print(f"验证码长度: {self.max_captcha}")
        print(f"字符类别数: {len(char_set)}")
        print(f"测试样本数: {total}\n")

    def gen_captcha_text_image(self):
        """随机获取验证码图片和标签"""
        img_name = random.choice(self.img_list)
        label = img_name.split("_")[0]
        img_file = os.path.join(self.img_path, img_name)
        return label, np.array(Image.open(img_file))

    def test_batch(self):
        """执行批量测试"""
        total_count = self.total
        correct_count = 0
        time_cost = 0

        for i in range(total_count):
            # 生成测试样本
            true_label, image_array = self.gen_captcha_text_image()

            # 预处理
            start_time = time.time()
            processed_image = self.convert2gray(image_array)
            processed_image = processed_image.flatten() / 255.0
            processed_image = np.expand_dims(processed_image, axis=0)  # 添加batch维度

            # 模型预测
            logits = self.predict(processed_image)
            predict_indices = tf.argmax(
                tf.reshape(logits, [-1, self.max_captcha, self.char_set_len]),
                axis=-1
            ).numpy()[0]

            # 解码预测结果
            predicted_label = ''.join([self.char_set[idx] for idx in predict_indices])
            time_cost += time.time() - start_time

            # 结果比对
            if true_label == predicted_label:
                correct_count += 1
                print(f"样本 {i + 1}/{total_count} ✓ 正确: {true_label} -> 预测: {predicted_label}")
            else:
                print(f"样本 {i + 1}/{total_count} ✗ 错误: {true_label} -> 预测: {predicted_label}")

        # 统计结果
        accuracy = correct_count / total_count
        avg_time = time_cost / total_count

        print("\n" + "=" * 30 + " 测试结果 " + "=" * 30)
        print(f"测试样本总数: {total_count}")
        print(f"正确识别数量: {correct_count}")
        print(f"识别准确率: {accuracy:.2%}")
        print(f"平均识别耗时: {avg_time:.4f}秒/张")
        print(f"总耗时: {time_cost:.2f}秒")


def main():
    # 加载配置文件
    with open("conf/sample_config.json") as f:
        config = json.load(f)

    # 配置参数
    test_config = {
        "test_image_dir": config["test_image_dir"],
        "model_save_dir": config["model_save_dir"],
        "char_set": json.load(open("tools/labels.json")) if config["use_labels_json_file"]
        else config["char_set"],
        "total": 100
    }

    # 执行测试
    try:
        tester = TestBatch(**test_config)
        tester.test_batch()
    except Exception as e:
        print(f"测试失败: {str(e)}")


if __name__ == '__main__':
    main()