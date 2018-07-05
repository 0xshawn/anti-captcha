import random
import string
import multiprocessing

import cv2
import matplotlib.pyplot as plt
import numpy as np

from captcha.image import ImageCaptcha
from bson import ObjectId



characters = string.digits + string.ascii_uppercase


def batch_generate_captcha_mat(count, cnt=1, w=30, h=30):
    capt = ImageCaptcha(width=cnt * 34 + 26, height=60)
    mat_x = np.ndarray((count, w, h), dtype=np.uint8)
    mat_y = np.zeros((count, corpus_len), dtype=np.uint8)
    for i in range(0, count):
        cid = random.randint(0, corpus_len - 1)
        c = corpus[cid]
        capt_img = np.array(capt.generate_image(c))
        gray_img = cv2.cvtColor(capt_img, cv2.COLOR_RGB2GRAY)
        scle_img = cv2.resize(gray_img, (w, h))
        mat_x[i] = scle_img
        mat_y[i][cid] = 1.0
    return mat_x, mat_y


def show_img(img, zoom=4, dpi=80):
    w = img.shape[0]
    h = img.shape[1]
    plt.figure(figsize=(w*zoom/dpi, h*zoom/dpi), dpi=dpi)
    plt.axis('off')
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()
    return

def batch_generate_captcha_to_file(corpus, batch=1, capt_len=1, w=30, h=30):
    capt = ImageCaptcha(width=capt_len * 30, height=60)
    for i in range(0, batch):
        chars = ''
        for j in range(0, capt_len):
            chars += random.choice(corpus)
        # image = capt.generate_image(chars)
        filename = 'data/{}_{}.png'.format(str(ObjectId()), chars)
        capt.write(chars, filename)

        
def generate_job(id):
    batch_generate_captcha_to_file(characters, batch=1, capt_len=4)


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    pool.map(generate_job, range(100000))
