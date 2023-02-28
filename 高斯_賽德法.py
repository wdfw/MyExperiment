#實驗名 : 高斯-賽德法
#目的 : 透過迭代法對矩陣方程式進行求解
#方法 : 以高斯-賽德法對矩陣方程求解,每次
#       迭代僅更新一個參數,且當參數的誤差
#       夠小,就不再更新此參數.並測試此方法
#       的收斂條件
#公式 : 以2維矩陣舉例 Ax = b, A,b are given ,find x
#       A = [[A11,A12],[A21,A22]] x = [x1,x2].T, b = [b1,b2].T, 
#       first iteration assume x1,x2 = 0, x1' = b1 - A12*x2 / A11,
#                       x2' = (b2 - A21*x1) / A22
#       並以此迭代下去

import numpy as np
# Ax ?= b 

size = 3 #n維矩陣
A = np.random.random([size,size])*10 #n*n 矩陣方程 
x = np.array([[0.0]*size]).T
b = np.random.random([size,1])*10 
ea = 1e-10 #相對誤差
#for i in range(size) : A[i,i] = (sum(A[i]) - A[i,i])*(np.random.random([1])+1)
#使用上行必定讓此高斯-賽德法收斂,否則高斯-賽德法不一定收斂,尤其維度越多時
stopFlag = np.array([False]*size)
flag_Cnt = 0 #當相對誤差夠小就不再更新 並升起flag

for j in range(200) : #最多迭代200次
    for i in range(size) :
        if stopFlag[i] : continue
        old_x,x[i,0] = x[i,0],0 
        x[i,0] = (b[i]-A[i].dot(x)) / A[i,i]
        if abs((old_x - x[i,0])/x[i,0]) < ea : stopFlag[i],flag_Cnt = True,flag_Cnt + 1
            
    if flag_Cnt ==  size: break
        
print("Truth Error \n" , A.dot(x) - b) #解的誤差
for i in range(size) : #測試是否符合收斂條件 Aii > sum(Aij) where (j from 0 to n-1) and (j != i) 
    x,y = A[i,i],sum(A[i]) - A[i,i]
    z = x > y
    print(x,y,z)
