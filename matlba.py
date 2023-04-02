#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/16 11:16
# @Author  : Xilun Wu
# @email   : nnuwxl@gmail.com
# @File    : sumPass.py.py
import numpy as np

def means_filter(input_image, filter_size):
    input_image_cp =np.copy(input_image)# 输入图像的副本
    filter_template = np.ones((filter_size, filter_size)) # 空间滤波器模板 卷积核
    # pad_num =int((filter_size - 1)/2) # 输入图像需要填充的尺寸
    pad_num =0 # 输入图像需要填充的尺寸
    input_image_cp = np.pad(input_image_cp,(pad_num, pad_num), mode="constant", constant_values=0) # 填充输入图像
    # m,n =input_image_cp.shape # 获取填充后的输入图像的大小
    m,n = 6,6
    output_image = np.copy(input_image_cp) # 输出图像
    output_image = np.array((6, 6))
    for j in range(pad_num, m - pad_num):
        for i in range(pad_num, n - pad_num):
    #卷积运算，并计算灰度均值返回到原来的像素点
            output_image[i, j] = np.sum(filter_template *input_image_cp[i- pad_num:i + pad_num + 1,j- pad_num: j+ pad_num + 1])/ (filter_size * 2)
            iutput_image = output_image[pad_num:m - pad_num, pad_num:n - pad_num] # 还原填充零之前的图像形状大小
    return output_image

def naiiveBoxFilter(im,r):
    H,W = im.shape
    H = H -2
    W = W -2
    res = np.zeros((H,W))
    for i in range(H):
        for j in range(W):
            s,n=0,0
            for k in range(i-r//2,i+r-r//2):
                for m in range(j-r//2,j+r-r//2):
                    if k<0 or k>=H or m<0 or m>=W:
                        continue
                    else:
                        s += im[k,m]
                        n += 1
            res[i,j] = s/n
    return res

if __name__ == "__main__":
    # 主函数
    img = [[2,7,6,1,3,6,9,5], [4,2,3,4,2,7,6,8],[8, 9, 6, 5, 3, 7, 3, 2], [6, 4, 5, 3, 2, 9, 4, 3], [5, 4 ,6 ,9, 4, 3, 7, 4], [ 3, 2, 4, 7, 5, 6, 3, 1],[4, 5, 6, 4, 3, 5, 7, 7], [1, 3, 5, 2, 4, 6, 8, 9]]
    img = np.array(img,dtype=float)
    # img = means_filter(img, 3)
    # img = naiiveBoxFilter(img, 3)
    nowarray=[]
    for rownum in range(len(img)):
        if rownum not in [0,len(img)-1]:
            nowrowlist = []
            for colnnum in range(len(img[0])):
                if colnnum not in [0,len(img[0])-1]:
                    nowlist = int((img[rownum-1][colnnum-1]+2*img[rownum-1][colnnum]+img[rownum-1][colnnum+1]+
                                   2*img[rownum][colnnum-1]+4*img[rownum][colnnum]+2*img[rownum][colnnum+1]+
                                   img[rownum+1][colnnum-1]+2*img[rownum+1][colnnum]+img[rownum+1][colnnum+1])/16)
                    nowrowlist.append(nowlist)
            nowarray.append(nowrowlist)

    img = np.array(nowarray,dtype = np.uint8)
    print(img)

