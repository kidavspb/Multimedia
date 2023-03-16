from PIL import Image
import numpy as np

# import struct
#
# bmp = open('kodim10.bmp', 'rb')
# print('bfType:', bmp.read(2).decode())
# print('bfSize: %s' % struct.unpack('I', bmp.read(4)))
# print('Reserved 1: %s' % struct.unpack('H', bmp.read(2)))
# print('Reserved 2: %s' % struct.unpack('H', bmp.read(2)))
# print('bfOffBits: %s' % struct.unpack('I', bmp.read(4)))
# print()
#
# print('biSize: %s' % struct.unpack('I', bmp.read(4)))
# print('biWidth: %s' % struct.unpack('I', bmp.read(4)))
# print('biHeight: %s' % struct.unpack('I', bmp.read(4)))
# print('biPlanes: %s' % struct.unpack('H', bmp.read(2)))
# print('biBitCount: %s' % struct.unpack('H', bmp.read(2)))
# print('biCompression: %s' % struct.unpack('I', bmp.read(4)))
# print('biSizeImage: %s' % struct.unpack('I', bmp.read(4)))
# print('biXPelsPerMeter: %s' % struct.unpack('I', bmp.read(4)))
# print('biYPelsPerMeter: %s' % struct.unpack('I', bmp.read(4)))
# print('biClrUsed: %s' % struct.unpack('I', bmp.read(4)))
# print('biClrImportant: %s' % struct.unpack('I', bmp.read(4)))

H = 768
W = 512

img = np.array(Image.open('kodim10.bmp'))


# print(img.shape)
# print(img)

def get_color_array(color):
    color_array = np.copy(img)
    for i in range(0, H):
        for j in range(0, W):
            for k in range(0, 3):
                if k == color: continue
                color_array[i][j][k] = 0
    return color_array


red_array = get_color_array(0)
green_array = get_color_array(1)
blue_array = get_color_array(2)

Image.fromarray(red_array).save('red_pic.bmp')
Image.fromarray(green_array).save('green_pic.bmp')
Image.fromarray(blue_array).save('blue_pic.bmp')


def variance(image, component):
    result = 0
    for i in range(0, H):
        for j in range(0, W):
            result += image[i][j][component]
    return result / (W * H)


def standard_deviation(image, component):
    var = variance(image, component)
    result = 0
    for i in range(0, H):
        for j in range(0, W):
            result += (image[i][j][component] - var) ** 2
    return (result / (W * H - 1)) ** 0.5


def correlation(image, componentA, componentB):
    varA = variance(image, componentA)
    varB = variance(image, componentB)
    tempArray = np.ndarray((H, W, 1))
    for i in range(0, H):
        for j in range(0, W):
            tempArray[i][j] = (image[i][j][componentA] - varA) * (image[i][j][componentB] - varB)
    varAB = variance(tempArray, 0)
    return varAB / (standard_deviation(image, componentA) * standard_deviation(image, componentB))


r_RG = correlation(img, 0, 1)
r_RB = correlation(img, 1, 2)
r_BG = correlation(img, 2, 0)
print(round(r_RG, 3), round(r_RB, 3), round(r_BG, 3))

def autocorrelation(image, component, x, y):
    var = variance(image, component)
    result = 0
    for i in range(0, H):
        for j in range(0, W):
            result += (image[i][j][component] - var) * (image[i + x][j + y][component] - var)
    return result / (W * H - 1)

print(autocorrelation(img, 0, 0, 0))
