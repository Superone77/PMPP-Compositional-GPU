import numpy as np
import matplotlib.pyplot as plt

def least_squares(x1,y1):
    x = np.array(x1)
    y = np.array(y1)
    x_ = x.mean()
    y_ = y.mean()
    m = np.zeros(1)
    n = np.zeros(1)
    k = np.zeros(1)
    p = np.zeros(1)

    for i in np.arange(6):
        k = (x[i]-x_)*(y[i]-y_)
        m+=k
        p = np.square(x[i]-x_)
        n = n+p
    a=m/n
    b = y_ -a*x_
    return a,b

x1 = [1,100,1000,3000,5000,10000]
y1 = [0.000494,0.0005459,0.008197,0.189,0.347,1.264]

f1 = np.poly1d(np.polyfit(x1,y1,2))
k1 = f1(x1)
c = f1(0)
x = f1(1)-c
y = f1(-1)-c
a = (x+y)/2
b = (x-y)/2


print("MD: a = ",a," b = " ,b, "c = ",c)


l1 = plt.plot(x1,y1,'r--',label='MD')
l_1 = plt.plot(x1,k1,'r-', lw=2,markersize=6)
plt.grid(b=True,ls=':')

plt.title('kernel line')
plt.xlabel('vector size')
plt.ylabel('time')
plt.legend()
plt.show()

x2 = [1,100,10000,65536,100000,1000000,3000000,5000000]
y11 = [0.0004214,0.0004342,0.0004548,0.0006180,0.0006294,0.003662,0.008365,0.0161]
y2 = [0.0004193,0.000441,0.0004543,0.0005930,0.0007083,0.002859,0.00994,0.0114]
y3 = [0.000456,0.0004553,0.0004905,0.0004944,0.001386,0.009247,0.0337,0.0498]
f2 = np.poly1d(np.polyfit(x2,y11,1))
f3 = np.poly1d(np.polyfit(x2,y2,1))
f4 = np.poly1d(np.polyfit(x2,y3,1))
k2 = f2(x2)
k3 = f3(x2)
k4 = f4(x2)
b = f2(0)
a = f2(1)-b
print("MIK: a = ",a," b = " ,b)
b = f3(0)
a = f3(1)-b
print("MAK: a = ",a," b = " ,b)
b = f4(0)
a = f4(1)-b
print("SPK: a = ",a," b = " ,b)

l2 = plt.plot(x2, y11, 'g--',label='MIK')
l3 = plt.plot(x2, y2, 'd--',label = 'MAK')
l4 = plt.plot(x2, y3, 'b--',label='SPK')
l_2 = plt.plot(x2,k2,'g-', lw=2,markersize=6)
l_3 = plt.plot(x2,k3,'d-', lw=2,markersize=6)
l_4 = plt.plot(x2,k4,'b-', lw=2,markersize=6)




# plt.plot(x1,y1,'ro-',x2,y11,'g+-',x2,y2,'d+-',x2,y3,'b^-')
plt.title('kernel line')
plt.xlabel('vector size')
plt.ylabel('time')
plt.legend()
plt.show()

print(f1(1500))
print(f2(8000000))
print(f3(60000)+0.000000606)
print(f4(200000)+0.000000606)
