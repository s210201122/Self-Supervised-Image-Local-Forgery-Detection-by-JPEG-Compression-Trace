import tensorflow as tf
import time
from model import detector
from glob import glob
import os
import cv2


def detector_test(detector, ckpt_dir, original_file, noise_file):
    original_files = glob(os.path.join(original_file, '*.png'))
    original_files.extend(glob(os.path.join(original_file, '*.TIF')))
    original_files.extend(glob(os.path.join(original_file, '*.jpg')))
    # original_files = sorted(original_files)
    start_time = time.time()
    detector.test(ckpt_dir, original_files, noise_file)

    end_time = time.time()
    print("[*] Test_My time:", end_time - start_time)


if __name__ == '__main__':
    data_path = './data/test_test/COCO_11'
    file_dir = os.listdir(data_path)
    for i in range(len(file_dir)):  # 对所有图像进行遍历
        print("This is the {} file:".format(i))
        filename = file_dir[i]
        img_path = os.path.join(data_path, filename)
        tamper_path = os.path.join(data_path, 'tamper')
        output_path = os.path.join(data_path, 'output/res')
        # res_output = os.path.join(output_path, '%04d' % (i))
        os.makedirs(output_path, exist_ok=True)

        ckpt_dir = './checkpoint'
        tf.reset_default_graph()
        with tf.Session() as sess:
            model = detector(sess)
            detector_test(model, ckpt_dir, tamper_path, output_path)


