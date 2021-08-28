import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import matplotlib.pyplot as plt


'''
Программа решает однородные квазилинейные НКЗ
Вся задача содержится в phi,mu,f и выборе констант сетки,
все остальное универсально
'''
S=100
N=S
J=S
x1=0
x2=-1
t1=0
t2=5
h=(x2-x1)/N #h<0 is allowed
tau=(t2-t1)/J
epsilon=1e-9

def phi(x): #initial conditions
    return 2-4/math.pi*math.atan(x+2)

def mu(t): #boundary conditions
    return (2-4/math.pi*math.atan(2))/math.exp(t)

def f(v,y_xnext,y_tnext): #for iterative solver
    return (v-y_xnext)/tau-(v*v-y_tnext*y_tnext)/(2*h)

def f_prime(v):
    return 1/tau-v/h

#main
y=np.zeros((N+1,J+1)) #solution

#boundary
for j in range(J+1):
    y[0][j]=mu(t1+j*tau)
#initial
for n in range(N+1):
    y[n][0]=phi(x1+n*h)

for j in range(J):
    for n in range(N):
        #find y^{j+1}_{n+1}
        #set initial value for iterations
        v_cur=y[n][j+1]
        v_next=v_cur-f(v_cur,y[n+1][j],y[n][j+1])/f_prime(v_cur)
        v_prev=0.5*v_cur
        q=0
        while (abs((v_next-v_cur)/(1-q))>epsilon):
            #update v
            v_prev=v_cur
            v_cur=v_next
            v_next=v_cur-f(v_cur,y[n+1][j],y[n][j+1])/f_prime(v_cur)
            #update q
            q=(v_next-v_cur)/(v_cur-v_prev)
        y[n+1][j+1]=v_next

#display y
plt.figure(1)
axes=plt.axes(projection="3d")

x = np.linspace(x1,x2,N+1)#endpoint=true
t = np.linspace(t1,t2,J+1)

Xcoord_matr, Ycoord_matr = np.meshgrid(x, t,indexing='ij')

axes.plot_surface(Xcoord_matr, Ycoord_matr, y, cmap='magma')
axes.set_xlabel('x')
axes.set_ylabel('t')
axes.view_init(elev=36, azim=-2)

#display y in countour plot
plt.figure(2)
plt.contour(Xcoord_matr, Ycoord_matr, y, levels=20, cmap='magma')
plt.grid()
plt.colorbar()
plt.title("solution countour plot")
plt.xlabel('x')
plt.ylabel('t')

#plot characterestics
plt.figure(3)
plt.grid()
t_star_array=np.linspace(t1,t2,20)
x_star_array=np.linspace(x1,x2,20)
x_axes=np.linspace(x1,x2,300)
for t_star in t_star_array:
    plt.plot(x_axes,t_star-1/mu(t_star)*x_axes)
for x_star in x_star_array:
    plt.plot(x_axes,-1/phi(x_star)*x_axes+x_star/phi(x_star))
plt.xlim(min(x1,x2),max(x1,x2))
plt.ylim(t1,t2)
plt.title("characterestics")
plt.xlabel('x')
plt.ylabel('t')
#show all plots
plt.show()