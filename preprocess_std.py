# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:53:02 2023

@author: user
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


def removeblack(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    black_mask = cv2.inRange(hsv_image, (0, 0, 0), (180, 255, 30))
    black_mask_inv = cv2.bitwise_not(black_mask)
    # plt.imshow(black_mask_inv)
    # plt.show()

    # 計算RGB頻率
    hist = cv2.calcHist([image], [0, 1, 2], black_mask_inv, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    
    # 找頻率最高的顏色
    most_frequent_color = np.unravel_index(np.argmax(hist), hist.shape)
    # print(most_frequent_color)
    
    # 取頻率最高的顏色
    result_img = np.zeros_like(image)
    result_img[:, :] = most_frequent_color
    
    # 代替黑色區域
    result_img[black_mask == 0] = image[black_mask == 0]
    return result_img, most_frequent_color
    # # 檢查是否存在黑色像素
    # has_black_pixel = np.any(np.all(result_img == [0, 0, 0], axis=-1))

    # # 輸出結果
    # if has_black_pixel:
    #     print("圖像中存在黑色像素")
    # else:
    #     print("圖像中不存在黑色像素")
    # plt.imshow(result_img)
    # plt.show()

def fill(im_in):
    """
    比較適合remote sencing
    """
    contours, _ = cv2.findContours(im_in, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # n = len(contours)  # 輪廓之個數
    cv_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 100000:
            # cv_contours.append(contour)
            cv2.drawContours(im_in, [contour], 0, (255, 255, 255), -1)
            # x, y, w, h = cv2.boundingRect(contour)
            # img[y:y + h, x:x + w] = 255
        else:
            continue
            
    im_out = cv2.fillPoly(im_in, cv_contours, (255, 255, 255))
    return im_out


def remove_small_points(src, threshold_area):
    '''
    去除白色的mask
    ----------
    src : 
        輸入的圖片矩陣
    threshold_area : int
        要去除的面積大小

    Returns
    -------
    img :
        去除小面積的mask
        
    '''
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(src, connectivity=8, ltype=None)
    img = np.zeros((src.shape[0], src.shape[1]), np.uint8)  
    for i in range(1, num_labels):
        mask = labels == i             
        if stats[i][4] > threshold_area:         
            img[mask] = 255
                      
        else:
            img[mask] = 0
           
    return img

def canny_process(img):
    """
    Cannt
    """
    # 先將圖片轉成灰階
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #設kernel給dilate and erosion
    kernel = np.ones((3,3),np.uint8) 
    
    #使用canny來找線條，因為陸地的線條相較於海面會很明顯
    #找出來之線條使用膨脹來加強
    # 1 20
    low_threshold = 1
    high_threshold = 20
    edges = cv2.Canny(gray_img, low_threshold, high_threshold)
    dilate = cv2.dilate(edges,kernel,iterations = 1)
    
    binary_image = remove_small_points(dilate, 70*70)  #去除船艦和零散區塊
    binary_image = cv2.bitwise_not(binary_image)
    binary_image = remove_small_points(binary_image, 70*70) #填補陸地
    binary_image = cv2.bitwise_not(binary_image)
    binary_image = cv2.erode(binary_image,kernel,iterations = 4)
    binary_image = cv2.dilate(binary_image,kernel,iterations = 3)
    binary_image = remove_small_points(binary_image, 70*70)  #去除船艦和零散區塊

    # import matplotlib
    # matplotlib.use('TkAgg')
    # plt.subplot(221), plt.imshow(img), plt.title('RGB')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(222), plt.imshow(edges, cmap='gray'), plt.title('edges')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(223), plt.imshow(dilate, cmap='gray'), plt.title('dilate')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(224), plt.imshow(binary_image, cmap='gray'), plt.title('binary_image')
    # plt.xticks([]), plt.yticks([])
    # plt.show()
    
    return binary_image

def standard_process(image):
    """
    計算圖片標準差
    """
    # 讀取影像
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    window_size = 5

    # 計算影像的行數和列數
    rows, cols = gray_img.shape
    row_indices = np.arange(rows - window_size + 1)
    col_indices = np.arange(cols - window_size + 1)
    
    window_shape = (window_size, window_size)
    window_strides = gray_img.strides
    windows = np.lib.stride_tricks.as_strided(
        gray_img,
        shape=(row_indices.size, col_indices.size) + window_shape,
        strides=(window_strides[0], window_strides[1]) + window_strides
    )
    
    # 計算局部統計方差
    variances = np.var(windows, axis=(2, 3))
    result = np.zeros((rows, cols), dtype=np.float32)
    
    # 將方差給對應位置
    result[row_indices[:, np.newaxis], col_indices] = variances
    thresholded_image = np.where(result > 15, 255, 0)
    
    # 正規化輸出才有值
    result_normalized = cv2.normalize(thresholded_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    binary_image2 = remove_small_points(result_normalized, 70*70)  #去除船艦和零散區塊
    binary_image2 = cv2.bitwise_not(binary_image2)
    binary_image2 = remove_small_points(binary_image2, 100*100) #填補陸地
    binary_image2 = cv2.bitwise_not(binary_image2)
    
    kernel = np.ones((3,3),np.uint8)
    final = cv2.erode(binary_image2,kernel,iterations = 1)    
    
    # plt.subplot(221), plt.imshow(image), plt.title('RGB')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(222), plt.imshow(result_normalized, cmap='gray'), plt.title('result_normalized')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(223), plt.imshow(binary_image2, cmap='gray'), plt.title('binary_image2')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(224), plt.imshow(final, cmap='gray'), plt.title('closing1')
    # plt.xticks([]), plt.yticks([])
    # plt.show()
    
    return final

def otsu_process(img):
    """
    對圖片進行OTSU來二值化
    """
    # gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret1, th1 = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    fil = fill(th1)
    res_img1 = remove_small_points(fil, 70*70)
    
    kernel = np.ones((3,3),np.uint8)
    opening1 = cv2.morphologyEx(res_img1, cv2.MORPH_OPEN, kernel, iterations=2)
    closing1 = cv2.morphologyEx(opening1, cv2.MORPH_CLOSE, kernel, iterations=2)
    fill_img1 = fill(closing1)
    
    return fill_img1


def hsv_process(img):
    max_value, min_value, max_pixel_values, hue_mean, hue_std, max_valley = hsv(img)
    max_pixel_value = max_pixel_values[0][0]
    
    # if max_pixel_values is None:
    #     max_pixel_values = max_value
    hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # print('max_pixel_values:',max_pixel_values)
    # print('max_value:',max_value)
    h_channel = hsv_image[:, :, 0]
    # threshold_value = (max_value + min_value) / 2 * 0.65
    threshold_value = int(max_valley)
    diff = abs(max_value - min_value)
    # print('diff hue: ',diff)
    # print('threshold_value: ',threshold_value)

    
    if diff <= 20:
        binary_image = np.zeros_like(h_channel)
    else:
        #要檢查是低光度圖片 還是高光度 來作為決定HSV的閥值
        if 0 < max_pixel_value < 70:
            # _, binary_image = cv2.threshold(h_channel, threshold_value, 255, cv2.THRESH_BINARY)
            # binary_image = cv2.bitwise_not(binary_image)
            if 70 <= max_value < 120:
                
                _, binary_image = cv2.threshold(h_channel, threshold_value, 255, cv2.THRESH_BINARY)
                binary_image = cv2.bitwise_not(binary_image)
            elif 120 <= max_value <= 180:
                _, binary_image = cv2.threshold(h_channel, threshold_value, 255, cv2.THRESH_BINARY)
            else:
                binary_image = np.zeros_like(h_channel)
                
        elif 70 <= max_pixel_value <= 180:
            if 70 <= max_value < 120:
                _, binary_image = cv2.threshold(h_channel, threshold_value, 255, cv2.THRESH_BINARY)
                binary_image = cv2.bitwise_not(binary_image)
            elif 120 <= max_value <= 180:
                # print(h_channel)
                _, binary_image = cv2.threshold(h_channel, threshold_value, 255, cv2.THRESH_BINARY)
                # print(binary_image)
                # plt.imshow(binary_image, cmap='gray')
                # plt.show()
            else:
                binary_image = np.zeros_like(h_channel)
        else:
            binary_image = np.zeros_like(h_channel)

    # _, binary_image = cv2.threshold(h_channel, threshold_value, 255, cv2.THRESH_BINARY)
    nonzero_count = np.count_nonzero(binary_image)
    total_pixels = img.shape[0] * img.shape[1]
    nonzero_ratio = nonzero_count / total_pixels
    
    if nonzero_ratio > 0.9:
        binary_image = np.zeros_like(h_channel)
        
    binary_image2 = cv2.bitwise_not(binary_image)
    binary_image2 = remove_small_points(binary_image2, 70*70)  #去除船艦和零散區塊
    
    binary_image2 = cv2.bitwise_not(binary_image2)
    binary_image2 = remove_small_points(binary_image2, 70*70) #填補陸地FHSV
    
    # plt.subplot(221), plt.imshow(img), plt.title('RGB')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(222), plt.imshow(hsv_image), plt.title('hsv_image')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(223), plt.imshow(binary_image, cmap='gray'), plt.title('binary_image')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(224), plt.imshow(binary_image2, cmap='gray'), plt.title('remove_image')
    # plt.xticks([]), plt.yticks([])
    # import matplotlib
    # matplotlib.use('TkAgg')
    # plt.show()
    
    return binary_image2, abs(max_value-min_value), hue_mean, hue_std

def hist(img):
    """
    輸入:圖片
    
    目的:計算圖片中的RGB三個通道之出現頻率最高之pixel value、
        其最高之頻率值以及pixel value是0~51所出現在圖片的占比
        
        
    """
    # print("--------------------",img.size)
    max_pixel_values = []  
    max_pixel_frequencies = []  
    pixel_frequencies = []
    RGB_img = img
    # RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    color = ['blue','springgreen','red'] 
    for i in [0,1,2]:
        #計算三個通道pixel values值方圖
        mask = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
        hist = cv2.calcHist([RGB_img], [i], mask, [256], [0.0, 255.0])
        # hist = cv2.calcHist([img],[i], None, [256], [0.0,255.0]) 
        
        #最高頻率之pixel value
        max_pixel_value = np.argmax(hist)
        max_pixel_frequency = np.max(hist)
        
        #最高頻率次數
        max_pixel_values.append(max_pixel_value)
        max_pixel_frequencies.append(max_pixel_frequency)
        
        #0~50的pixel頻率出現比值
        pixel_frequency = np.sum(hist[1:50]) / np.sum(hist)
        pixel_frequencies.append(pixel_frequency)
        
    #     plt.subplot(121), plt.plot(hist, color[i])
    #     plt.title('Histrogram of Color image')

    # plt.subplot(122), plt.imshow(RGB_img), plt.title('res_img1')
    # plt.xticks([]), plt.yticks([])
    # plt.tight_layout()
    
    # for i, channel in enumerate(color):
    #     print(f"Max pixel value in {channel} channel: {max_pixel_values[i]}")
    #     print(f"Frequency of max pixel value in {channel} channel: {max_pixel_frequencies[i]}")
    #     print(f"Frequency of pixels in {channel} channel between 0 and 50: {pixel_frequencies[i]}")

    # import matplotlib
    # matplotlib.use('TkAgg')
    # plt.show()
    return max_pixel_values, max_pixel_frequencies, pixel_frequencies

def labcolor(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    histA = np.zeros(256)
    histB = np.zeros(256)
    msqA = 0
    msqB = 0
    L_channel = lab_image[:, :, 0]
    A_channel = lab_image[:, :, 1]
    B_channel = lab_image[:, :, 2]
    # print(f'L: {np.std(L_channel)},A: {np.std(A_channel)-128},B: {np.std(B_channel)-128}')

    h, w, _ = image.shape

    # 將範圍移動至[-128, 127] 因為是opencv
    da = A_channel.mean() - 128
    db = B_channel.mean() - 128
    # print('da: ',da)
    # print('db: ',db)

    # 遮罩出非黑色像素
    non_black_mask = np.any(image != [0, 0, 0], axis=-1)

    # 計算直方圖 
    # hist是計算 A和B值 出現的次數
    np.add.at(histA, A_channel[non_black_mask].flatten(), 1) 
    np.add.at(histB, B_channel[non_black_mask].flatten(), 1)

    non_black_pixels = np.sum(non_black_mask)

    # 計算 msqA 和 msqB (Mean Squared)
    # 將範圍移動至[-128, 127] 因為是opencv
    # 得到每個 像素值 與 均值的差
    # hist是-128~127 每一個像素值 出現的機率
    # histA / (w * h) 為出現的機率
    y = np.arange(256)
    msqA = np.sum(np.abs(y - 128 - da) * histA) / non_black_pixels
    msqB = np.sum(np.abs(y - 128 - db) * histB) / non_black_pixels
    if msqA ==0 and msqB ==0:
        result = 0
    elif msqA == 0:
        result = math.sqrt(da**2 + db**2) / math.sqrt(msqB**2)
    elif msqB == 0:
        result = math.sqrt(da**2 + db**2) / math.sqrt(msqA**2)
    else:
        result = math.sqrt(da**2 + db**2) / math.sqrt(msqA**2 + msqB**2)

    # 計算 d/m 值
    # print('d/m:', result)
    return result



def hsv(img):
    """
    判斷HSV中的H通道範圍
    """
    freq_values = []  
    hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # 提取H通道
    h_channel = hsv_image[:, :, 0]
    s_channel = hsv_image[:, :, 1]
    v_channel = hsv_image[:, :, 2]

    # 計算H通道非0的值
    mask = np.where(v_channel == 0, 0, 255).astype(np.uint8)
    hist = cv2.calcHist([h_channel], [0], mask, [256], [0, 256])
    hist2 = hist.copy()

    peaks = np.where((hist[:-2] < hist[1:-1]) & (hist[1:-1] > hist[2:]))[0] + 1
    # valleys = np.where((hist[:-2] > hist[1:-1]) & (hist[1:-1] < hist[2:]))[0] + 1

    # 找到最高峰值和次高峰值
    sorted_peaks = np.argsort(hist[peaks].flatten())
    max_peak_idx = peaks[sorted_peaks[-1]]
    if len(sorted_peaks) >= 2:
        second_peak_idx = peaks[sorted_peaks[-2]]
        # 兩個峰值之間的區域
        if max_peak_idx > second_peak_idx:
            valley_region = hist[second_peak_idx:max_peak_idx]
        else:
            valley_region = hist[max_peak_idx:second_peak_idx]

        # 尋找峰谷索引
        min_valley_idx = np.argmin(valley_region)

        # 最大峰谷索引
        if max_peak_idx > second_peak_idx:
            max_valley_idx = second_peak_idx + min_valley_idx
        else:
            max_valley_idx = max_peak_idx + min_valley_idx
    else:
        max_valley_idx = max_peak_idx

    # sorted_valleys = np.argsort(hist[valleys].flatten())
    # max_valley_idx = peaks[sorted_valleys[-1]]



    # # # 顯示直方圖
    # plt.figure(figsize=(10, 5))
    # plt.plot(hist, color='blue')
    # plt.scatter(max_peak_idx, hist[max_peak_idx], color='red', label='Max Peak')
    # plt.scatter(second_peak_idx, hist[second_peak_idx], color='green', label='Second Peak')
    # plt.scatter(max_valley_idx, hist[max_valley_idx], color='blue', label='Valley')

    # plt.xlabel('Hue Value')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Hue Channel (V Channel Masked)')
    # plt.legend()
    # plt.show()

    # plt.bar(range(256),hist.flatten())
    # plt.show()

    max_freq_value = np.argmax(hist)
    hist2[max_freq_value] = 0
    second_max_freq_value = np.argmax(hist2)
    # print('max_freq_value: ',max_freq_value)
    # print('max_pixel_value: ',second_max_freq_value)

    freq_values.append((max_freq_value,second_max_freq_value))
    # print('freq_values: ',freq_values)

    
    # # 黑色像素的遮罩
    # black_mask = (s_channel == 0 ) & (v_channel == 0)
    
    # # 将遮罩应用于提取黑色像素
    # black_pixels = hsv_image[black_mask]
    # print(black_pixels.shape)
    # h_channel = black_pixels

    # 計算H通道的平均值和標準差
    hue_mean = np.mean(h_channel)
    hue_std = np.std(h_channel)
        
    # 找頻率超過某數的值
    high_freq_values = np.where(hist > 1000)[0]
    
    if len(high_freq_values) == 0:
        # print("No pixel values with frequency greater than 1000.")
        # 找到值方中的最大值和最小值
        # hue_mean = np.mean(h_channel)
        # hue_std = np.std(h_channel)

        max_value = np.max(hist)
        min_value = np.min(hist)
        
    else:
        # 在高頻率像素值中找到最大值和最小值
        max_value = None
        min_value = None

        max_value = np.max(high_freq_values)
        min_value = np.min(high_freq_values)
        
        max_value = max_value if max_value is not None else min_value
        min_value = min_value if min_value is not None else max_value
        # hue_mean = np.mean(h_channel)
        # hue_std = np.std(h_channel)

    # print(hue_std)
    
    return max_value, min_value, freq_values, hue_mean, hue_std, max_valley_idx

def final_process(img, lab_val=0):
    '''
    最終影像mask合併
    '''
    # lab_val = 1

    hsv, color_range, hue_mean, hue_std = hsv_process(img) #Hue
    standard = standard_process(img) #標準差
    canny = canny_process(img) #線條

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    kernel = np.ones((3,3),np.uint8)
    output1 = cv2.add(standard, canny) #合併
    output1 = cv2.erode(output1, kernel, iterations=1)
    output2 = hsv
    output2 = cv2.erode(output2, kernel, iterations=1)


    #尋找不是0的像素有幾個
    # nonzero_count1 = np.count_nonzero(output1) #Canny+Stnadard
    nonzero_count2 = np.count_nonzero(output2) #HSV
    
    #計算mask比例
    total_pixels = img.shape[0] * img.shape[1]
    # nonzero_ratio1 = nonzero_count1 / total_pixels #Canny+Stnadard
    nonzero_ratio2 = nonzero_count2 / total_pixels #HSV
    
    #計算mask比例
    total_pixels = img.shape[0] * img.shape[1]
    # nonzero_ratio1 = nonzero_count1 / total_pixels #Canny+Stnadard
    nonzero_ratio2 = nonzero_count2 / total_pixels #HSV
    # print('nonzero_ratio2: ',nonzero_ratio2)
    # print('hue std: ',hue_std)
    if nonzero_ratio2 == 1 and hue_std > 10:
        otsu = otsu_process(img)
        output2 = otsu

    # output = cv2.add(output1, output2) #合併
    output = output1
    # output_no = output
    output = cv2.bitwise_not(output)
    output1 = cv2.bitwise_not(cv2.add(standard, canny)) #合併
    output2 = cv2.bitwise_not(hsv)
    if nonzero_ratio2 == 1 and hue_std > 10:
        output2 = cv2.bitwise_not(otsu)

    img2 = img.copy()
    
    zero_ratio = np.count_nonzero(output == 0) / (img.shape[0] * img.shape[1])
    # plt.subplot(2, 2, 1)
    # plt.title('hue_std:'+str(int(hue_std)))
    # plt.imshow(img2)
    # plt.axis('off')

    # plt.subplot(2, 2, 2)
    # plt.title('Output Mask: '+str(int(zero_ratio*100))+' %')
    # plt.imshow(cv2.bitwise_not(output), cmap='gray')
    # plt.axis('off')

    # plt.subplot(2, 2, 3)
    # plt.title('Std+Canny')
    # plt.imshow(output1, cmap='gray')
    # plt.axis('off')

    # plt.subplot(2, 2, 4)
    # plt.title('HSV')
    # plt.imshow(output2, cmap='gray')
    # plt.axis('off')

    # # plt.savefig(f'D:\\yolov7_2\\pdr\\sim\\sealand\\{int(hue_std)+(random_number)}.png')
    # import matplotlib
    # matplotlib.use('TkAgg')
    # plt.show()
    # plt.close()
    
    return output, color_range, hue_mean, hue_std


def ANGS(img, s_factors = 7, e_lambda = 1e-7):
    """

    Parameters
    ----------
    img : array
       Input image
    s_factors : int
        The default is 7.
    e_lambda : float
        prevent 0

    Returns
    -------
    normalized_img : array
        Output image

    """
    # print(img)
    # img = cv2.imread(img)
    
    img_float = img.astype(np.float32)
    
    channel_means = np.mean(img_float, axis=(0, 1))
    
    angs_weights = (channel_means / (img_float+e_lambda))**s_factors
    normalized_img = 1/(1+angs_weights)
    # print(normalized_img)
    normalized_img = np.clip(normalized_img*255, 0, 255).astype(np.uint8)
    
    # print(normalized_img)
    return normalized_img 
                
def yoloPreProcess(img,dir_path,filename):

    sealandmode = True
    ANGSmode = False
    sfactor = 1

    max_pixel_values, max_pixel_frequencies, pixel_frequencies = hist(img)
    
    #blue, green, red
    BMP, GMP, RMP = max_pixel_values
    RMF = max_pixel_frequencies[2]
    RF = pixel_frequencies[2]

    raw_img = img.copy()
    if ANGSmode and not sealandmode:
        _hsv, _color_range, _hue_mean, hue_std = hsv_process(img,savehsv=False)
        return ANGS(img, sfactor) if hue_std >= 12 or RMP <= GMP <= BMP else raw_img
    
    elif sealandmode and not ANGSmode:
        output, _color_range, _hue_mean, hue_std = final_process(img)
        zero_ratio = np.count_nonzero(output == 0) / (raw_img.shape[0] * raw_img.shape[1])
        print(f'{filename} hue_std: {hue_std}')
        if cv2.countNonZero(output) == 0:
            output = cv2.bitwise_not(output)
        return cv2.bitwise_and(raw_img, raw_img, mask=output)
        if hue_std >= 15 and zero_ratio >= 0.9: #都是陸地
            print(f'All Land {zero_ratio}')
            return raw_img
        else:
            if hue_std >= 15 and zero_ratio < 0.9:
                print(f'Sea and Land {zero_ratio}')
                return cv2.bitwise_and(raw_img, raw_img, mask=output)
            else:
                print(f'Sea {zero_ratio}')
                return raw_img
            # return cv2.bitwise_and(raw_img, raw_img, mask=output) if hue_std >= 12 and zero_ratio < 0.9 else raw_img
            

                    
    elif sealandmode and ANGSmode:
        # _hsv, _color_range, _hue_mean, hue_std = hsv_process(img,savehsv=False)
        # ANGS_output = ANGS(img, sfactor) if hue_std >= 12 or RMP <= GMP <= BMP else raw_img
        # output, _color_range, _hue_mean, hue_std = final_process(ANGS_output,dir_path,filename,savefinal=True)
        
        output = final_process(raw_img,dir_path,filename,max_pixel_values,sfactor,ANGSmode=True,savefinal=True)
        # img, dir_path, filename, max_pixel_values, sfactor, ANGSmode, savefinal=False
        return output
        # zero_ratio = np.count_nonzero(output == 0) / (raw_img.shape[0] * raw_img.shape[1])

        # if cv2.countNonZero(output) == 0:
        #     output = cv2.bitwise_not(output)

        # if hue_std >= 12 and zero_ratio >= 0.9:
        #     return ANGS_output
        # else:
        #     return cv2.bitwise_and(raw_img, raw_img, mask=output) if RMP <= GMP and RMP <= BMP and zero_ratio < 0.9 else ANGS_output
            # return ANGS_output if RMP <= GMP and RMP <= BMP and zero_ratio < 0.9 else raw_img

    else:
        return raw_img          
    

def imgProcess(filePath):
    #只要是以下副檔名都可以被讀取
    extensions = tuple(['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo'])
    
    for filename in tqdm(os.listdir(filePath), desc="Sea Land"):
        
        if filename.endswith(extensions):
            # print(filename)
            # dir_path = os.path.dirname(filePath) + "/Sealand_lilium_512_5"
            dir_path = os.path.dirname(filePath) + "/sealand_test4"
            
            if os.path.exists(dir_path) == False:
                os.makedirs(dir_path)

            img = cv2.imread(filePath + "/" + filename)
            print(img.shape)
            output_img = yoloPreProcess(img,dir_path,filename)
            # print(dir_path + "/" + filename)
            cv2.imwrite(dir_path + "/" + filename, output_img)
            

if __name__ == "__main__":
    # path = "D:\\Sealand\\ship\\test\\images"
    # path = "D:\\Sealand\\ship\\test\\cloud_lilium_slice\\Sealand_lilium_512"
    path = "D:\\yolov7_2\\attention-test\\test4"
    # path ='D:\\yolov7_2\\moving_object_0726_selected'
    imgProcess(path)
    

    
    print('done')