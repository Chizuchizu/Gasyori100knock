import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\ochim\PycharmProjects\dl\Gasyori100knock\Question_21_30\imori.jpg")


def normal(x, c, d, a=0, b=255):
    if x < c:
        return a
    elif d < x:
        return b
    else:
        return (b-a)/(d-c) * (x-c)


def normal22(x, s, m, s0, m0):
    return s0 / (s*(x-m)) + m0


img2 = np.zeros((132, 132, 3))
c = np.min(img2)
d = np.max(img2)
tmp = np.zeros_like(img2)
s = np.std(img2)
m = np.mean(img2)

# out = d / (128*128*3)*
# out[out < 0] = 0
# out[out > 255] = 255
# sum_h = 0
# for i in range(1, 255):
#     ind = np.where(img2 == i)
#     sum_h += len(img[ind])
#     Z_prime = 255 / (128*128*3) * sum_h
#     out[ind] = Z_prime
#
# out = out.astype(np.uint8)

def zscore(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return (x-np.min(x)) / (np.max(x) - np.min(x))

# パディング
for i in range(128):
    for j in range(128):
        for k in range(3):
            img2[i+1, j+1, k] = img[i, j, k]


x = y = int(128*1.5)
out = np.zeros((x, y, 3)).astype(np.uint8)


def h(t):
    if abs(t) <= 1:
        return abs(t)**3 - 2*(abs(t)**2) + 1
    elif 2 < abs(t):
        return 0
    else:
        return -abs(t)**3 - (-5*abs(t)**2) + -8*abs(t) + 4


for i in range(x):
    for j in range(x):
        for k in range(3):
            x_prime = int(np.floor(i/1.5))
            y_prime = int(np.floor(j/1.5))

            dx1 = i / 1.5 - (x_prime-1)
            dx2 = i / 1.5 - x_prime
            dx3 = (x_prime+1) - i / 1.5
            dx4 = (x_prime+2) - i / 1.5
            dy1 = j / 1.5 - (y_prime - 1)
            dy2 = j/1.5 - y_prime
            dy3 = (y_prime+1) - j/1.5
            dy4 = (y_prime+2) - j/1.5

            wxi = [h(dx1), h(dx2), h(dx3), h(dx4)]
            wyi = [h(dy1), h(dy2), h(dy3), h(dy4)]
            print(wxi)
            o = 0
            o1 = 0
            for a in range(-1, 3):
                for b in range(-1, 3):
                    o += img2[x_prime+a+1, y_prime+b+1, k] * wxi[a+1] * wyi[b+1]
                    o1 += wxi[a+1] * wyi[b+1]
            print(o, o1)
            if o == 0.0 and o1 == 0.0:
                out[i, j, k] = 0
            else:
                out[i, j, k] = int(o / o1)


# for i in range(128):
#     for j in range(128):
#         for k in range(3):
#             o = normal(img2[i, j, k], s, m, 52, 128)
#             tmp[i, j, k] = o


# plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
# plt.show()
# plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
# plt.show()
out = np.where(out < 0, 0, out)
out[255 < out] = 255
out = out.astype(np.uint8)
# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)
