'''
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
path = os.path.abspath(os.path.dirname(sys.argv[0]))

img = plt.imread(path+'\\FFT\\1.png')
plt.subplot(231),plt.imshow(img),plt.title('picture')
print(img.shape)
#根据公式转成灰度图
img = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
print(img.shape)
#显示灰度图
plt.subplot(232),plt.imshow(img,'gray'),plt.title('original')
 
#进行傅立叶变换，并显示结果
fft2 = np.fft.fft2(img)
plt.subplot(233),plt.imshow(np.abs(fft2),'gray'),plt.title('fft2')
 
#将图像变换的原点移动到频域矩形的中心，并显示效果
shift2center = np.fft.fftshift(fft2)
plt.subplot(234),plt.imshow(np.abs(shift2center),'gray'),plt.title('shift2center')
 
#对傅立叶变换的结果进行对数变换，并显示效果
log_fft2 = np.log(1 + np.abs(fft2))
plt.subplot(235),plt.imshow(log_fft2,'gray'),plt.title('log_fft2')
 
#对中心化后的结果进行对数变换，并显示结果
log_shift2center = np.log(1 + np.abs(shift2center))
plt.subplot(236),plt.imshow(log_shift2center,'gray'),plt.title('log_shift2center')
plt.show()
'''
#-------------------------------------------------------------------------根据公式实现的二维离散傅立叶变换如下
## 超级慢的计算，所以不会用到
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

path = os.path.abspath(os.path.dirname(sys.argv[0]))
img = plt.imread(path+'\\FFT\\3.png')

PI = 3.141591265
 
#根据公式转成灰度图
img = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
 
#显示原图
plt.subplot(131),plt.imshow(img,'gray'),plt.title('original')
 
#进行傅立叶变换，并显示结果
fft2 = np.fft.fft2(img)
log_fft2 = np.log(1 + np.abs(fft2))
plt.subplot(132),plt.imshow(log_fft2,'gray'),plt.title('log_fft2')
 
h , w = img.shape
#生成一个同样大小的复数矩阵
F = np.zeros([h,w],'complex128')
for u in range(h):
    for v in range(w):
        res = 0
        for x in range(h):
            for y in range(w):
                res += img[x,y] * np.exp(-1.j * 2 * PI * (u * x / h + v * y / w))
        F[u,v] = res
log_F = np.log(1 + np.abs(F))
plt.subplot(133),plt.imshow(log_F,'gray'),plt.title('log_F')
plt.show()


#--------------------------------------opencv实现傅里叶变换：
'''
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
path = os.path.abspath(os.path.dirname(sys.argv[0]))
img = cv2.imread(path+'\\FFT\\1.png',0)
 
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
 
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
 
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
'''

