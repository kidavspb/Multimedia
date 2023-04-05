# import numpy as np
# from PIL import Image
from imports import *

R = Y = 0
G = Cb = 1
B = Cr = 2


def get_color_array(image, color):
    H = image.shape[0]
    W = image.shape[1]
    color_array = np.copy(image)
    for i in range(0, H):
        for j in range(0, W):
            for k in range(0, 3):
                if k == color: continue
                color_array[i][j][k] = 0
    return color_array


def variance(image, component):
    H = image.shape[0]
    W = image.shape[1]
    result = 0
    for i in range(0, H):
        for j in range(0, W):
            result += image[i][j][component]
    return result / (W * H)


def standard_deviation(image, component):
    H = image.shape[0]
    W = image.shape[1]
    var = variance(image, component)
    result = 0
    for i in range(0, H):
        for j in range(0, W):
            result += (image[i][j][component] - var) ** 2
    return (result / (W * H - 1)) ** 0.5


def correlation(image, componentA, componentB):
    H = image.shape[0]
    W = image.shape[1]
    varA = variance(image, componentA)
    varB = variance(image, componentB)
    tempArray = np.ndarray((H, W, 1))
    for i in range(0, H):
        for j in range(0, W):
            tempArray[i][j] = (image[i][j][componentA] - varA) * (image[i][j][componentB] - varB)
    varAB = variance(tempArray, 0)
    return varAB / (standard_deviation(image, componentA) * standard_deviation(image, componentB))


def autocorrelation(image, component, y, x):
    H = image.shape[0]
    W = image.shape[1]
    if y >= 0 and x >= 0:
        img1 = image[:H - y, :W - x]
        img2 = image[y:, x:]
    elif y <= 0 and x <= 0:
        y = abs(y)
        x = abs(x)
        img1 = image[y:, x:]
        img2 = image[:H - y, :W - x]
    elif y <= 0 and x >= 0:
        y = abs(y)
        img1 = image[y:, :W - x]
        img2 = image[:H - y, x:]
    elif y >= 0 and x <= 0:
        x = abs(x)
        img1 = image[:H - y, x:]
        img2 = image[y:, :W - x]

    H -= y
    W -= x
    var1 = variance(img1, component)
    var2 = variance(img2, component)
    tempArray = np.ndarray((H, W, 1))
    for i in range(0, H):
        for j in range(0, W):
            tempArray[i][j] = (img1[i][j][component] - var1) * (img2[i][j][component] - var2)
    var = variance(tempArray, 0)
    return var / (standard_deviation(img1, component) * standard_deviation(img2, component))


def RGB_to_YCbCr(image):
    H = image.shape[0]
    W = image.shape[1]
    YCbCR_image = np.copy(image)
    for i in range(H):
        for j in range(W):
            YCbCR_image[i][j][Y] = 0.299 * image[i][j][R] + 0.587 * image[i][j][G] + 0.114 * image[i][j][B]
            YCbCR_image[i][j][Cb] = 0.5643 * (int(image[i][j][B]) - int(YCbCR_image[i][j][Y])) + 128
            YCbCR_image[i][j][Cr] = 0.7132 * (int(image[i][j][R]) - int(YCbCR_image[i][j][Y])) + 128
    return YCbCR_image


def replicate_to_all_channels(image, component):
    H = image.shape[0]
    W = image.shape[1]
    result = np.copy(image)
    for i in range(H):
        for j in range(W):
            for k in range(0, 3):
                result[i][j][k] = image[i][j][component]
    return result


def YCbCr_to_RGB(image):
    H = image.shape[0]
    W = image.shape[1]
    RGB_image = np.copy(image)
    for i in range(H):
        for j in range(W):
            RGB_image[i][j][R] = Sat(image[i][j][Y] + 1.402 * (image[i][j][Cr] - 128))
            RGB_image[i][j][G] = Sat(image[i][j][Y] - 0.714 * (image[i][j][Cr] - 128) - 0.334 * (image[i][j][Cb] - 128))
            RGB_image[i][j][B] = Sat(image[i][j][Y] + 1.772 * (image[i][j][Cb] - 128))
    return RGB_image


def Sat(x, xmin=0, xmax=255):
    if x < xmin:
        return xmin
    elif x > xmax:
        return xmax
    else:
        return x


def PSNR(image1, image2, component):
    H = image1.shape[0]
    W = image1.shape[1]
    L = 8
    MSE = 0
    for i in range(H):
        for j in range(W):
            MSE += (image1[i][j][component] - image2[i][j][component]) ** 2
    MSE /= (H * W * 3)
    return 10 * np.log10(W * H * (2 ** L - 1) ** 2 / MSE)


def decimate_odd_rows_cols(YCbCR_image):
    H = YCbCR_image.shape[0]
    W = YCbCR_image.shape[1]
    for i in range(H):
        for j in range(W):
            if not (i % 2 == 0 and j % 2 == 0):
                YCbCR_image[i][j][Cb] = 0
                YCbCR_image[i][j][Cr] = 0
    return YCbCR_image


def decimate_mean(YCbCR_image):
    H = YCbCR_image.shape[0]
    W = YCbCR_image.shape[1]
    for i in range(H, 2):
        for j in range(W, 2):
            average_Cb = 0
            average_Cr = 0
            for k in range(2):
                for l in range(2):
                    average_Cb += YCbCR_image[i + k][j + l][Cb]
                    average_Cr += YCbCR_image[i + k][j + l][Cr]
                    YCbCR_image[i + k][j + l][Cb] = 0
                    YCbCR_image[i + k][j + l][Cr] = 0
            YCbCR_image[i][j][Cb] = average_Cb//4
            YCbCR_image[i][j][Cr] = average_Cr//4
    return YCbCR_image
