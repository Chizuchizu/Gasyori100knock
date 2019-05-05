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


img2 = np.zeros((130, 130, 3))
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

# # パディング
# for i in range(128):
#     for j in range(128):
#         for k in range(3):
#             img2[i+1, j+1, k] = img[i, j, k]


# out = np.zeros((128, 128, 3))
img2 = img.copy()


def h(t):
    if abs(t) <= 1:
        return abs(t)**3 - 2*(abs(t)**2) + 1
    elif 2 < abs(t):
        return 0
    else:
        return -abs(t)**3 - (-5*abs(t)**2) + -8*abs(t) + 4


x_prime = int(128 * 1.3)
y_prime = int(128 * 0.8)
out = np.zeros((y_prime, x_prime, 3))


tx = -30
ty = 30

for i in range(128):  # 128x1.5
    for j in range(128):
        for k in range(3):
            x = i
            y = j
            K = np.array([[1.3, 0, tx],
                          [0, 0.8, ty],
                          [0, 0, 1]])
            o = np.array([[x],
                          [y],
                          [1]])
            k_o = np.dot(K, o)
            x_prime = int(k_o[0])
            y_prime = int(k_o[1])
            print(x_prime, y_prime)

            if x_prime < 0 or y_prime < 0 or x_prime >= 98 or y_prime >= 158:
                break
            else:
                print(x_prime, y_prime)
                out[x_prime, y_prime, k] = img2[i, j, k]


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
out[out > 255] = 255
out = out.astype(np.uint8)
# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)
