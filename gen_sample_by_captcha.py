# -*- coding: UTF-8 -*-
"""
使用captcha lib生成验证码（前提：pip install captcha）
"""
# -*- coding: UTF-8 -*-
from captcha.image import ImageCaptcha
import os
import random
import json
from concurrent.futures import ThreadPoolExecutor


class CaptchaGenerator:
    def __init__(self, width, height):
        self.generator = ImageCaptcha(width=width, height=height)

    def gen_special_img(self, text, file_path):
        img = self.generator.generate_image(text)
        img.save(file_path)


def generate_random_text(characters, char_count):
    return ''.join(random.choices(characters, k=char_count))


def gen_ima_by_batch(tread, root_dir, image_suffix, characters, count, char_count, width, height):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    generator = CaptchaGenerator(width, height)
    texts = [generate_random_text(characters, char_count) for _ in range(count)]

    with ThreadPoolExecutor(max_workers=tread) as executor:
        futures = []
        for i, text in enumerate(texts):
            p = f"{root_dir}/{text}_{i}.{image_suffix}"
            futures.append(executor.submit(generator.gen_special_img, text, p))

        for idx, future in enumerate(futures):
            future.result()
            if idx % 1000 == 0:
                print(f"Generated {idx + 1}/{count} images")


# main函数保持原结构不变


def main():
    with open("conf/captcha_config.json", "r") as f:
        config = json.load(f)
    # 配置参数
    root_dir = config["root_dir"]  # 图片储存路径
    image_suffix = config["image_suffix"]  # 图片储存后缀
    characters = config["characters"]  # 图片上显示的字符集 # characters = "0123456789abcdefghijklmnopqrstuvwxyz"
    count = config["count"]  # 生成多少张样本
    char_count = config["char_count"]  # 图片上的字符数量
    tread = config["tread"] #  使用的线程数
    # 设置图片高度和宽度
    width = config["width"]
    height = config["height"]

    gen_ima_by_batch(tread, root_dir, image_suffix, characters, count, char_count, width, height)


if __name__ == '__main__':
    main()