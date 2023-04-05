from utils import *

img = np.array(Image.open('kodim10.bmp'))
H = img.shape[0]
W = img.shape[1]


def task3():
    red_array = get_color_array(img, R)
    green_array = get_color_array(img, G)
    blue_array = get_color_array(img, B)

    Image.fromarray(red_array).save('red_pic.bmp')
    Image.fromarray(green_array).save('green_pic.bmp')
    Image.fromarray(blue_array).save('blue_pic.bmp')


def task4(img=img):
    task4a(img)
    task4b(img)


def task4a(img=img):
    r_RG = correlation(img, R, G)
    r_RB = correlation(img, R, B)
    r_BG = correlation(img, B, G)

    # print(round(r_RG, 2), round(r_RB, 2), round(r_BG, 2))
    print("Correlation between R ang G: ", r_RG)
    print("Correlation between R ang B: ", r_RB)
    print("Correlation between B ang G: ", r_BG)


def task4b(img=img):
    auto_red = []
    auto_green = []
    auto_blue = []
    for y in range(-10, 11, 5):
        auto_red.append([autocorrelation(img, R, x, y) for x in range(-10, 11, 5)])
        auto_green.append([autocorrelation(img, G, x, y) for x in range(-10, 11, 5)])
        auto_blue.append([autocorrelation(img, B, x, y) for x in range(-10, 11, 5)])

    plt.plot([x for x in range(-10, 11, 5)], auto_red)
    plt.show()
    plt.plot([x for x in range(-10, 11, 5)], auto_green)
    plt.show()
    plt.plot([x for x in range(-10, 11, 5)], auto_blue)
    plt.show()


def task5():
    YCbCR_image = RGB_to_YCbCr(img)
    task4(YCbCR_image)


def task6():
    YCbCR_image = RGB_to_YCbCr(img)

    Y_img = replicate_to_all_channels(YCbCR_image, Y)
    Cb_img = replicate_to_all_channels(YCbCR_image, Cb)
    Cr_img = replicate_to_all_channels(YCbCR_image, Cr)

    Image.fromarray(Y_img).save('Y_pic.bmp')
    Image.fromarray(Cb_img).save('Cb_pic.bmp')
    Image.fromarray(Cr_img).save('Cr_pic.bmp')


def task7():
    YCbCR_image = RGB_to_YCbCr(img)
    RGB_image = YCbCr_to_RGB(YCbCR_image)

    Image.fromarray(RGB_image).save('restored_RGB.bmp')

    PSNR_R = PSNR(img, RGB_image, R)
    PSNR_G = PSNR(img, RGB_image, G)
    PSNR_B = PSNR(img, RGB_image, B)

    print("PSNR for R component: ", PSNR_R)
    print("PSNR for G component: ", PSNR_G)
    print("PSNR for B component: ", PSNR_B)


def task8():
    task8a()
    task8b()


def task8a():
    RGB_image = YCbCr_to_RGB(decimate_odd_rows_cols(RGB_to_YCbCr(img)))
    # Image.fromarray(RGB_image).save('(8a)restored_RGB.bmp')


def task8b():
    RGB_image = YCbCr_to_RGB(decimate_mean(RGB_to_YCbCr(img)))
    # Image.fromarray(RGB_image).save('(8b)restored_RGB.bmp')

def task9():
    YCbCR_image = decimate_mean(RGB_to_YCbCr(img))
    for i in range(H):
        for j in range(W):
            if not (i % 2 == 0 and j % 2 == 0):
                YCbCR_image[i][j][Cb] = 0
                YCbCR_image[i][j][Cr] = 0