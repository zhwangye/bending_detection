from ctypes import *
import math
import random
import os
import cv2 as cv
import time
import numpy as np
import shutil
from dropper import utils
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import data, filters, segmentation, measure, morphology, color,io,transform
import skimage
from skimage import morphology
import loose.utils


def lstsq_bending(center_points):
    """最小二乘法拟合直线，计算残差"""
    center_points = np.asarray(center_points)
    len_cpt = len(center_points)
    # 去除异常点x异常点
    center_points = center_points[center_points[:, 0].argsort()][int(len_cpt * 0.3):int(len_cpt * 0.95)]
    # 去除y异常点
    len_cpt = len(center_points)
    center_points = center_points[center_points[:, 1].argsort()][int(len_cpt * 0.2):int(len_cpt * 0.95)]

    x, y = center_points[..., 0], center_points[..., 1]
    A = np.vstack([x, np.ones(len(x))]).T

    # plt.plot(x,y)
    # plt.show()

    if len(list(A.shape)) >= 2:
        root = np.linalg.lstsq(A, y, rcond=-1)
    if root is not None:
        if len(x) > 0:
            # res = str(root[1][0]) + "  " + str(len(x)) + "  " + str(root[1][0] / len(x))
            if len(root[1])==1:     # 残差不为0
                flag = root[1][0] / len(x)
                return flag
            else:
                return 0
    else:
        return 0

def projection(binary):
    '''对二值图像先进行距离变换，然后再进行水平和垂直投影
        目的是得到图像的骨架
    @param
    binary: 二值图片
    @return:
    result: cv.image 二值骨架图片
    xptx: 垂直投影的点
    ypts: 水平投影的的
    '''
    h, w = binary.shape
    dist = cv.distanceTransform(binary, cv.DIST_L1, cv.DIST_MASK_PRECISE)
    result = np.zeros((h, w), dtype=np.uint8)
    ypts = []
    for row in range(h):
        cx = 0
        cy = 0
        max_d = 0
        for col in range(w):
            d = dist[row][col]
            if d >= max_d:
                max_d = d
                cx = col
                cy = row
        result[cy][cx] = 255
        ypts.append([cx, cy])

    xpts = []
    for col in range(w):
        cx = 0
        cy = 0
        max_d = 0
        for row in range(h):
            d = dist[row][col]
            if d >= max_d:
                max_d = d
                cx = col
                cy = row
        result[cy][cx] = 255
        xpts.append([cx, cy])
    return result,xpts,ypts


def line_fit(frame):
    '''
    获取距离变换，水平投影和垂直投影后的骨架图
    @param frame: cv.img color
    @return:
    nres: 骨架图，gray
    data: 图像中前景坐标
    '''
    if len(frame.shape)==2:
        frame = cv.cvtColor(frame,cv.COLOR_GRAY2BGR)

    # 二值化图像
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray,(5,5),0)
    ret, binary = cv.threshold(gray, 3, 255, cv.THRESH_BINARY)

    # 距离变换，投影
    result,xpts,ypts = projection(binary)

    # 每一行前景如果像素值多于1，则该行置为0
    cnt = np.sum(result,-1)
    idx = np.where(cnt>255)
    result[idx, :] = 0
    space = 15
    result = result[space:-space,space:-space]

    ## 去除离散点
    result = result >1
    result = morphology.remove_small_objects(result, min_size=2, connectivity=1)

    ## 布尔图还原到灰度图
    data = []
    nres = np.zeros(result.shape,dtype=np.uint8)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if result[i][j] == False:
                nres[i][j]=0
            else:
                nres[i][j]=255
                data.append([i,j])

    return nres,data

def rr_img(img):
    '''旋转和去除离群点'''
    debug = False
    img = loose.utils.get_rotate_img(img)
    if debug:
        cv.imshow('src_img', img)
    # 计算图片吊弦所在的x坐标
    if len(img) > 0:
        h, w = img.shape
        center = []
        for j in range(h):
            row = []
            for i in range(w):
                if img[j][i] == 255:
                    row.append(i)
            if len(row) > 0:
                center.append(np.mean(row))
        center_x = np.mean(center).astype(np.int)

        # 截取吊弦两侧blank像素的范围
        blank = 20
        for j in range(h):
            for i in range(0, center_x - blank):
                img[j][i] = 0
            for i in range(center_x + blank, w):
                img[j][i] = 0

        if debug: cv.imshow('blank', img)
        # # 去除离散点
        result = img > 1
        result = morphology.remove_small_objects(result, min_size=20, connectivity=1)

        # bool 图片到常规图片
        data = []
        nres = np.zeros(result.shape, dtype=np.uint8)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if result[i][j] == False:
                    nres[i][j] = 0
                else:
                    nres[i][j] = 255
                    data.append([i, j])
        if debug: cv.imshow('rm oulier', nres)
        return nres

def dropper_main(img):
    '''检测主函数'''
    debug = False
    is_wrong = False
    if img is not None:
        simg = cv.resize(img, (160, 480))
        if len(list(simg.shape)) >2:
            img = cv.cvtColor(simg,cv.COLOR_BGR2GRAY)
        else:
            img = simg
        if debug: cv.imshow('src_gr',img)
        # img, _ = line_fit(img)    # 拟合直线
        if debug:
            cv.imshow('t1', img)
            cv.waitKey(0)
        # img = rr_img(img)     # 旋转图片
        if img is not None:
            img, data = line_fit(img)
            if img is not None:
                # simg = loose.utils.get_rotate_img(simg)
                if len(data) != 0:
                    th_res = lstsq_bending(data)
                    print('bending: ', th_res)
                    if th_res is None:
                        th_res = 0
                    if 0.7 <th_res < 30:
                        is_wrong = True
                        # cv.imshow('ttt', simg)
                        # cv.waitKey(0)
                    else:
                        is_wrong = False

    if debug:
        cv.imshow('ttt', img)
        cv.waitKey(0)
    return is_wrong


if __name__ == "__main__":
    import multiprocessing
    stime = time.time()
    xdir = "E:\\07d\\test\\r1"

    abdir = [os.path.join(xdir,name) for name in os.listdir(xdir)]

    pool = multiprocessing.Pool(processes=12)
    pool.map(mutil,abdir)
    pool.close()
    pool.join()

    # for i,dir in enumerate(abdir):
    #     print(dir)
    #     # dir = 'E:\\darkenet_detection\\z01_merge_cls\\dr_v1\\231605136_K896285_183_2_29_2.jpg'
    #     dir = 'E:\\darkenet_detection\\z01_merge_cls\\dr_v1\\000000220_K862139_430_2_29_1.jpg'
    #     img = cv.imread(dir,cv.IMREAD_COLOR)
    #     img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #     flag = dropper_main(img)
    #     if flag:
    #         shutil.copy(dir,dir.replace('dr_v1','dr_z0'))
    #     else:
    #         shutil.copy(dir, dir.replace('dr_v1', 'dr_z1'))

    print('Time: ',time.time() - stime)
