import numpy as np
import tensorflow as tf
import os
import cv2


def extractor(input, is_training=True, output_channels=1):
    with tf.variable_scope('block1'):
        output = tf.layers.conv2d(input, 64, 3, padding='same', activation=tf.nn.relu)
    for layers in range(2, 16+1):
        with tf.variable_scope('block%d' % layers):
            output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=False)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
    with tf.variable_scope('block17'):
        output = tf.layers.conv2d(output, output_channels, 3, padding='same', use_bias=False)
    return output, input - output


class detector(object):
    def __init__(self, sess, input_c_dim=1, batch_size=128):
        self.sess = sess
        self.input_c_dim = input_c_dim
        self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim], name='clean_image')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.X = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim])
        self.R, self.Y = extractor(self.X, is_training=self.is_training)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0

    def test(self, ckpt_dir, original_files, noise_file):
        """Test"""
        tf.global_variables_initializer().run()  # 初始化变量
        load_model_status, global_step = self.load(ckpt_dir)  # 读入模型参数
        assert load_model_status == True, '[!] Load weights FAILED...'
        print("[*] Load weights SUCCESS...")
        for i in range(len(original_files)):
            original = cv2.imread(original_files[i])
            dtype = np.float32
            original = np.asarray(original).astype(dtype)
            original = (0.299 * original[:, :, 0] + 0.587 * original[:, :, 1] + 0.114 * original[:, :, 2]) / 255.0
            img = original.copy()
            h, w = original.shape
            original = original[np.newaxis, ..., np.newaxis]  # 增加维度[w, h, 1]-->[1, w, h, 1]
            slide = 1024  # 3072
            largeLimit = 1050000  # 9437184 大于1024x1024就会达到最大限制
            overlap = 34
            output_residual = np.zeros((h, w), np.float32)
            output_clean = np.zeros((h, w), np.float32)
            if h * w <= largeLimit:
                output_residual, output_clean = self.sess.run(
                    [self.R, self.Y], feed_dict={self.X: original, self.is_training: False})
            else:
                for index0 in range(0, h, slide):
                    index0start = index0 - overlap
                    index0end = index0 + slide + overlap

                    for index1 in range(0, w, slide):
                        index1start = index1 - overlap
                        index1end = index1 + slide + overlap
                        clip = img[max(index0start, 0): min(index0end, h), \
                               max(index1start, 0): min(index1end, w)]
                        clip = clip[np.newaxis, :, :, np.newaxis]
                        resB = self.sess.run(self.R, feed_dict={self.X: clip, self.is_training: False})
                        resB = np.squeeze(resB)
                        if index0 > 0:
                            resB = resB[overlap:, :]
                        if index1 > 0:
                            resB = resB[:, overlap:]
                        resB = resB[:min(slide, resB.shape[0]), :min(slide, resB.shape[1])]

                        output_residual[index0: min(index0 + slide, h), \
                        index1: min(index1 + slide, w)] = resB
            residual = np.squeeze(output_residual)

            vmin = np.min(residual[34:-34, 34:-34])
            vmax = np.max(residual[34:-34, 34:-34])
            np.clip(residual, vmin, vmax)
            residual = (255 * (residual.clip(vmin, vmax) - vmin) / (vmax - vmin)).clip(0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(noise_file, '%04d.png' % (i)), residual)
            # clean_file_path = 'E:'
            # output_clean = np.squeeze(output_clean)
            # output_clean = np.clip(255 * output_clean, 0, 255).astype('uint8')
            # cv2.imwrite(os.path.join(clean_file_path, '%d.png' % (i)), output_clean)
        print("[*] --- Test ----")