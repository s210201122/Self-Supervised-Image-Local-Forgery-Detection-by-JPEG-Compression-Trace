import cv2
import os
import numpy as np
import sklearn.metrics as metrics



def cal_prf(pred, gt):
    '''prf'''
    pred = np.where(pred > 127, 255, 0)
    pred = pred.reshape(-1)
    gt = np.where(gt == 0, 0, 255)
    gt = gt.reshape(-1)
    pred_neg = 255 - pred
    P_o = metrics.precision_score(gt, pred, pos_label=255, average='binary')
    R_o = metrics.recall_score(gt, pred, pos_label=255, average='binary')
    F_o = metrics.f1_score(gt, pred, pos_label=255, average='binary')
    P_n = metrics.precision_score(gt, pred_neg, pos_label=255, average='binary')
    R_n = metrics.recall_score(gt, pred_neg, pos_label=255, average='binary')
    F_n = metrics.f1_score(gt, pred_neg, pos_label=255, average='binary')
    if F_o > F_n:
        return P_o, R_o, F_o
    else:
        return P_n, R_n, F_n


if __name__ == '__main__':
    file_path = "./data/test"
    file_dir = os.listdir(file_path)
    for file in range(0, len(file_dir)):
        file_name = os.path.join(file_path, file_dir[file])
        results_dir = os.listdir(file_name)
        gts_path = os.path.join(file_name, 'tamper_gt')
        for ret in range(4, len(results_dir)-5):
            results_path = os.path.join(file_name, results_dir[ret])
            outputs_path = os.path.join(results_path, 'binary')
            binary_dir = os.listdir(outputs_path)
            gt_dir = os.listdir(gts_path)
            PRF_path = os.path.join(results_path, 'PRF')
            os.makedirs(PRF_path, exist_ok=True)
            F1_file = os.path.join(PRF_path, 'F.txt')
            P_file = os.path.join(PRF_path, 'P.txt')
            R_file = os.path.join(PRF_path, 'R.txt')
            F1 = open(F1_file, 'w')
            P = open(P_file, 'w')
            R = open(R_file, 'w')
            F_sum = 0
            P_sum = 0
            R_sum = 0
            for i in range(len(binary_dir)):  # len(imgs_dir)
                binary_path = os.path.join(outputs_path, binary_dir[i])
                gt_path = os.path.join(gts_path, gt_dir[i])
                binary = cv2.imread(binary_path)
                gt = cv2.imread(gt_path)
                p, r, f = cal_prf(binary, gt)
                F_sum += f
                P_sum += p
                R_sum += r
                print("P, R, F = {}, {}, {}".format(p, r, f))
                F1.write("\n{}".format(f))
                P.write("\n{}".format(p))
                R.write("\n{}".format(r))
            print("---------------------------------")
            F1.write("\n-----------\n{}".format(F_sum / len(binary_dir)))
            P.write("\n-----------\n{}".format(P_sum / len(binary_dir)))
            R.write("\n-----------\n{}".format(R_sum / len(binary_dir)))
            print("Average P: {}".format(P_sum / len(binary_dir)))
            print("Average R: {}".format(R_sum / len(binary_dir)))
            print("Average F1: {}".format(F_sum / len(binary_dir)))
            F1.close()
            P.close()
            R.close()