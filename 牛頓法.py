#實驗名 : 牛頓法找根
#目的 : 透過牛頓法的方式找出函數的根
#方法 : 使用兩種方法進行,第一種透過手動微分進行牛頓法,第二種用有
#       限差分自動進行牛頓法,並設定差的閥值,當函數與0的差小於閥
#       值或找不到時則停止
#公式 : y = f(x), find x let y = 0, x_k+1 = x_k - f(x_k) / f'(x_k)
import numpy as np
import matplotlib.pyplot as plt
import os
def Draw_Line(startP,endP,rate,offset = 0) :
    X = np.array([startP,endP])
    Y = rate*X + offset - rate*(startP+endP)/2
    plt.plot(X,Y,color = "Red")
    
def fun(x) : return x**2 - 0.5 #X的函數
def der(x) : return 2*x #函數的微分

size = 100 #擬合的點
domain = 3 #X輸入範圍


X = np.linspace(-domain,domain,num = size)
Y = fun(X)

P = -2 #起始點
e = 3e-4 #與0的閥值
delt = 1e-5 #差分的距離
it = 0

while abs(fun(P)) > e and it < 1000:
    print(P,abs(fun(P)))
    Draw_Line(P-0.5,P+0.5,der(P),fun(P))
    #P -=  fun(P) / der(P) #微分法
    P -= (fun(P)*delt) / (fun(P+delt) - fun(P))#差分法
     
    it += 1

plt.plot(X,Y)
plt.show()    

