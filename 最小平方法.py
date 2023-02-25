#實驗名 : 最小平方法在二維之擬合
#目的 : 透過線性迴歸的方式找出資料分布的狀態
#方法 : 以 Y = AX + B + N 之式子做分布的模擬,X代表輸入 A,B為常數 N為噪音 Y為輸出
#       透過最小平方法Y_pre = AX + B的方式擬合真實的Y值
#公式 : Y_pre = AX + B 以矩陣表示為Y_pre = W*X.T , X = [1 x] W = [B A]
#       得到 W = (X*X.T)^-1 * X * Y
import numpy as np
import matplotlib.pyplot as plt

def fun(x) : return -2*x + 3 #X無噪音下的輸出
size = 50 #擬合的點
domain = 1000 #X輸入範圍
noise = 10 #噪音範圍

X = (np.random.rand(size)*domain).reshape(-1,1) # 10,1
Base = np.ones(size).reshape(-1,1) # 10,1
N = np.random.normal(domain,noise,size=size).reshape(-1,1) #10,1
Y = fun(X)+N #10,1

plt.scatter(X,Y)

X_base = np.concatenate([Base,X],axis = 1)  # 10,1 -> 10,2

W = np.linalg.inv(X_base.T.dot(X_base)).dot(X_base.T).dot(Y).reshape(1,2) # 2,0

Base = np.ones(domain).reshape(1,domain) # 1,10
X_in = np.arange(0,domain).reshape(1,domain) #1,10
X_in = np.concatenate([Base,X_in],axis = 0)#2,10
Y_pre = W.dot(X_in).reshape(-1)

plt.plot(np.arange(0,domain),Y_pre)
plt.show()
