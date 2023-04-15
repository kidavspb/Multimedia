import os

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


def task8a(times=2):
    return decimate_odd_rows_cols(RGB_to_YCbCr(img), times)


def task8b(times=2):
    return decimate_mean(RGB_to_YCbCr(img), times)


def task9(times=2):
    YCbCR_image = task8a(times)
    # YCbCR_image = task8b(times)
    for i in range(0, H, times):
        for j in range(0, W, times):
            for k in range(times):
                for l in range(times):
                    YCbCR_image[i + k][j + l][Cb] = YCbCR_image[i][j][Cb]
                    YCbCR_image[i + k][j + l][Cr] = YCbCR_image[i][j][Cr]
    return YCbCr_to_RGB(YCbCR_image)


def task10(times=2):
    initial_RGB_image = np.copy(img)
    initial_YCbCR_image = RGB_to_YCbCr(initial_RGB_image)
    restored_RGB_image = task9(times)
    restored_YCbCR_image = RGB_to_YCbCr(restored_RGB_image)

    PSNR_Cb = PSNR(initial_YCbCR_image, restored_YCbCR_image, Cb)
    PSNR_Cr = PSNR(initial_YCbCR_image, restored_YCbCR_image, Cr)
    print("PSNR for Cb component: ", PSNR_Cb)
    print("PSNR for Cr component: ", PSNR_Cr)

    PSNR_R = PSNR(initial_RGB_image, restored_RGB_image, R)
    PSNR_G = PSNR(initial_RGB_image, restored_RGB_image, G)
    PSNR_B = PSNR(initial_RGB_image, restored_RGB_image, B)
    print("PSNR for R component: ", PSNR_R)
    print("PSNR for G component: ", PSNR_G)
    print("PSNR for B component: ", PSNR_B)

    Image.fromarray(restored_RGB_image).save('restored_RGB_pic.bmp')


def task11():
    task10(4)


def task12():
    YCbCr_image = RGB_to_YCbCr(img)

    R_freq = img[:, :, R].flatten()
    G_freq = img[:, :, G].flatten()
    B_freq = img[:, :, B].flatten()
    Y_freq = YCbCr_image[:, :, Y].flatten()
    Cb_freq = YCbCr_image[:, :, Cb].flatten()
    Cr_freq = YCbCr_image[:, :, Cr].flatten()

    plt.hist(R_freq, bins=256, color='red')
    plt.show()
    plt.hist(G_freq, bins=256, color='green')
    plt.show()
    plt.hist(B_freq, bins=256, color='blue')
    plt.show()
    plt.hist(Y_freq, bins=256, color='yellow')
    plt.show()
    plt.hist(Cb_freq, bins=256, color='cyan')
    plt.show()
    plt.hist(Cr_freq, bins=256, color='magenta')
    plt.show()


def task13():
    nx_R, nx_G, nx_B = np.zeros(256), np.zeros(256), np.zeros(256)
    nx_Y, nx_Cb, nx_Cr = np.zeros(256), np.zeros(256), np.zeros(256)
    YCbCr_image = RGB_to_YCbCr(img)

    for i in range(H):
        for j in range(W):
            nx_R[img[i][j][R]] += 1
            nx_G[img[i][j][G]] += 1
            nx_B[img[i][j][B]] += 1
            nx_Y[YCbCr_image[i][j][Y]] += 1
            nx_Cb[YCbCr_image[i][j][Cb]] += 1
            nx_Cr[YCbCr_image[i][j][Cr]] += 1

    n = H * W
    H_R = entropy(nx_R / n)
    H_G = entropy(nx_G / n)
    H_B = entropy(nx_B / n)
    H_Y = entropy(nx_Y / n)
    H_Cb = entropy(nx_Cb / n)
    H_Cr = entropy(nx_Cr / n)

    print("Entropy for R component: ", H_R)
    print("Entropy for G component: ", H_G)
    print("Entropy for B component: ", H_B)
    print("Entropy for Y component: ", H_Y)
    print("Entropy for Cb component: ", H_Cb)
    print("Entropy for Cr component: ", H_Cr)


def task14():
    YCbCr_image = RGB_to_YCbCr(img)

    d_array_RGB = np.copy(img)
    d_array_YCbCr = np.copy(YCbCr_image)

    rule = 1

    for i in range(1, H):
        for j in range(1, W):
            for k in range(3):
                d_array_RGB[i - 1][j - 1][k] = int(img[i][j][k]) - int(get_prediction_pixel(rule, img, i, j, k))
                d_array_YCbCr[i - 1][j - 1][k] = int(YCbCr_image[i][j][k]) - int(
                    get_prediction_pixel(rule, YCbCr_image, i, j, k))

    return d_array_RGB[0:H - 1, 0:W - 1], d_array_YCbCr[0:H - 1, 0:W - 1]


def task15():
    d_array_RGB, d_array_YCbCr = task14()

    R_freq = d_array_RGB[:, :, R].flatten()
    G_freq = d_array_RGB[:, :, G].flatten()
    B_freq = d_array_RGB[:, :, B].flatten()
    Y_freq = d_array_YCbCr[:, :, Y].flatten()
    Cb_freq = d_array_YCbCr[:, :, Cb].flatten()
    Cr_freq = d_array_YCbCr[:, :, Cr].flatten()

    plt.hist(R_freq, bins=256, color='red')
    plt.show()
    plt.hist(G_freq, bins=256, color='green')
    plt.show()
    plt.hist(B_freq, bins=256, color='blue')
    plt.show()
    plt.hist(Y_freq, bins=256, color='yellow')
    plt.show()
    plt.hist(Cb_freq, bins=256, color='cyan')
    plt.show()
    plt.hist(Cr_freq, bins=256, color='magenta')
    plt.show()


def task16():
    nx_R, nx_G, nx_B = np.zeros(256), np.zeros(256), np.zeros(256)
    nx_Y, nx_Cb, nx_Cr = np.zeros(256), np.zeros(256), np.zeros(256)
    d_array_RGB, d_array_YCbCr = task14()
    H = d_array_RGB.shape[0]
    W = d_array_RGB.shape[1]

    for i in range(H):
        for j in range(W):
            nx_R[d_array_RGB[i][j][R]] += 1
            nx_G[d_array_RGB[i][j][G]] += 1
            nx_B[d_array_RGB[i][j][B]] += 1
            nx_Y[d_array_YCbCr[i][j][Y]] += 1
            nx_Cb[d_array_YCbCr[i][j][Cb]] += 1
            nx_Cr[d_array_YCbCr[i][j][Cr]] += 1

    n = H * W
    H_R = entropy(nx_R / n)
    H_G = entropy(nx_G / n)
    H_B = entropy(nx_B / n)
    H_Y = entropy(nx_Y / n)
    H_Cb = entropy(nx_Cb / n)
    H_Cr = entropy(nx_Cr / n)

    print("Entropy for R component: ", H_R)
    print("Entropy for G component: ", H_G)
    print("Entropy for B component: ", H_B)
    print("Entropy for Y component: ", H_Y)
    print("Entropy for Cb component: ", H_Cb)
    print("Entropy for Cr component: ", H_Cr)


def task17():
    dop3a()


def dop3a():
    YCbCr_image = RGB_to_YCbCr(img)
    bitonic_planes = []
    for i in range(8):
        bitonic_planes.append((YCbCr_image[:, :, Y] >> i) & 1)

    for i in range(8):
        bitonic_plane = Image.fromarray(bitonic_planes[i].astype(np.uint8) * 255)
        bitonic_plane.save(os.path.join(os.getcwd(), f"bitonic_plane_{i}.bmp"))