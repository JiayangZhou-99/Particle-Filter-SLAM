#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 17:40:04 2022

@author: zhoujiayang
"""
import numpy as np
import matplotlib
import cv2
import matplotlib.pyplot as plt;



    
    
if __name__ == '__main__':
    for i in range (1,12):
        map = np.load("mapprob_data_{}.npy".format(i*10000))
        path = np.load("map_data_{}.npy".format(i*10000))
        new_map = np.ones((map.shape[0],map.shape[1],3))
        for i in range (0,map.shape[0]):
            for j in range(0,map.shape[1]):
                if map[i,j] >= 0:
                    new_map[i,j,:] = [0,0,0]
                if path[i,j] == 2:
                    new_map[i,j,:] = [1,0,0]
        plt.imshow(new_map)
        plt.show()
    map = np.load("mapprob_data.npy")
    path = np.load("map_data.npy")
    new_map = np.ones((map.shape[0],map.shape[1],3))
    for i in range (0,map.shape[0]):
        for j in range(0,map.shape[1]):
            if map[i,j] >= 0:
                new_map[i,j] = [0,0,0]
            if path[i,j] == 2:
                new_map[i,j,:] = [1,0,0]
    plt.imshow(new_map)
    plt.show()
    
    
    # map = np.load("map_data.npy")
    # img = np.zeros((map.shape[0],map.shape[1],3))
    # for i in range (0,map.shape[0]):
    #     for j in range (0,map.shape[1]):
    #         if map[i,j] == 1:
    #             img[i,j] = np.array([1,0,0])
    #         if map[i,j] == 2:
    #             img[i,j] = np.array([0,1,0])
    # plt.imshow(img)
    # plt.show()
    # # matplotlib.image.imsave('name.jpg', img)
    
    # plt.contour(map[::-1,:])
    
    
    # ret,thresh = cv2.threshold(img,127,255,0)
    # image ,contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # #绘制独立轮廓，如第四个轮廓
    # imag = cv2.drawContours(img,contours,-1,(0,255,0),3)
    # #但是大多数时候，下面方法更有用
    # #imag = cv2.drawContours(img,contours,3,(0,255,0),3)
    
    # while(1):
    #     cv2.imshow('img',img)
    #     cv2.imshow('image',image)
    #     cv2.imshow('imag',imag)
    #     if cv2.waitKey(1) == ord('q'):
    #         break
    # cv2.destroyAllWindows()
    
    # img = cv2.imread("name.jpg", cv2.IMREAD_UNCHANGED)

    # 二值化
    # ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY),1,255,cv2.THRESH_BINARY)  # 黑白二值化
    
    # # 搜索轮廓
    # contours, hierarchy = cv2.findContours(
    #     thresh,
    #     cv2.RETR_EXTERNAL,
    #     cv2.CHAIN_APPROX_SIMPLE)
    
    # # 画出轮廓
    # cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
    # cv2.imshow("contours", img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
