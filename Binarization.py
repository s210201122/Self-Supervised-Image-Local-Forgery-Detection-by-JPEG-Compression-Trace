import cv2
import numpy as np
import os


def gamma_trans(img, gamma):  # gamma函数处理
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。


if __name__ == '__main__':
    file_path = "./data/test"
    file_dir = os.listdir(file_path)
    for file in range(len(file_dir)):
        file_name = os.path.join(file_path, file_dir[file])
        results_dir = os.listdir(file_name)
        for ret in range(0, len(results_dir)-2):
            results_path = os.path.join(file_name, results_dir[ret])
            res_path = os.path.join(results_path, 'res')
            res_dir = os.listdir(res_path)
            outputs_path = os.path.join(results_path, 'binary')
            os.makedirs(outputs_path, exist_ok=True)
            post_processing_path = os.path.join(results_path, 'post_processing')
            os.makedirs(post_processing_path, exist_ok=True)

            for i in range(len(res_dir)):  # len(imgs_dir)
                img_path = os.path.join(res_path, res_dir[i])
                img = cv2.imread(img_path)
                h, w = img.shape[:2]
                # kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
                kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                # kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                dft = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
                fshift = np.fft.fftshift(dft)
                rows, cols = img_gray.shape
                crow, ccol = int(rows/2), int(cols/2)
                mask = np.zeros((rows, cols, 2), np.uint8)
                mask[crow-50:crow+50, ccol-50:ccol+50] = 1
                f = fshift * mask
                ishift = np.fft.ifftshift(f)
                iimg = cv2.idft(ishift)
                res = cv2.magnitude(iimg[:,:,0], iimg[:,:,1])
                res = (255 * (res - np.min(res)) / (np.max(res) - np.min(res))).clip(0, 255).astype(np.uint8)
                # cv2.imshow('res', res)
                # res = cv2.resize(res, (int(w / 5), int(h / 5)))
                res = 255 - res
                res = gamma_trans(res, 5.0)
                cv2.imwrite(os.path.join(post_processing_path, res_dir[i]), res)
                img = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
                # binary = kmeans(img)
                ret_otsu, binary = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                #binary = cv2.medianBlur(binary, 2)
                binary = cv2.erode(binary, kernel3, iterations=2)
                binary = cv2.dilate(binary, kernel3, iterations=2)
                output_path = os.path.join(outputs_path, res_dir[i])
                cv2.imwrite(output_path, binary)
                # cv2.imshow('alpha-image', binary)
                # cv2.waitKey()