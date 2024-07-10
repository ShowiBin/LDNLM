from LDNLM import ATTNLM
import time
import scipy.io as sio
import torch
import numpy as np
import matplotlib.pyplot as plt

def f_attnlm(noisy):
    t1 = time.time()
    noisy = torch.from_numpy(noisy)
    res= attnlm.denoise(noisy,batch_size=16)
    t2 = time.time()
    # return res,t2-t1
    res_img = res[0]
    res_img[res_img > 1] = 1
    res_img[res_img < 0] = 0
    return res_img,t2-t1
def get_min_max(S):
    s = S[S != 10*np.log10(np.finfo(float).eps)]  # 将S矩阵转换为一个列矩阵

    med0 = np.median(s)  # 取整个矩阵的中位值
    med1 = med0
    med2 = med0

    # 优化最大、最小值
    for m in range(3):  # 取左区间的中位值，循环取3次，当最小值
        temp1 = s[s < med1]
        if temp1.size == 0:
            break
        med1 = np.median(temp1)

    for m in range(3):  # 取右区间的中位值，循环取3次，当最大值
        temp2 = s[s > med2]
        if temp2.size == 0:
            break
        med2 = np.median(temp2)
    
    xMin = med1
    xMax = med2
    return xMin, xMax
def img256(img):
    img = (img - img.min())/(img.max()-img.min())
    img = (img * 255).astype(np.int64)
    return img


nei_size = 36
kernel_size = 9
device = 'cuda:0'
attnlm = ATTNLM(nei_size=nei_size,kernel_size=kernel_size,nhead = 8,nlayer=2,pos_cod_len=18000,model_path='./LDNLM_best.pt')



# Inference 
d = sio.loadmat('./decorr_complex_tsx_SLC_0.mat')
img_ = d['cout']

img_ = np.abs(img_)
# 进行截断处理
v_min,v_max = get_min_max(img_)
img_[img_<=v_min] = v_min
img_[img_>=v_max] = v_max

img_ = img256(img_)
img = img_[:256,512:512+256]
img = ((np.float32(img)+1.0)/256.0)


res,_ = f_attnlm(img[:512,:512])


plt.imshow((res)*255,'gray',vmin=0,vmax=255)
plt.show()


