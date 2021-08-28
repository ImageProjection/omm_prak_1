'''
программа решает начально-краевую задачу
для двумерного уравнения теплопроводности
с однородными ГУ Дирихле и однородным НУ
'''

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

N=60
M=60
J=100
l_x=np.pi
l_y=3
T=2

h_x=l_x/N
h_y=l_y/M
tau=T/J

def trimatrix_solver(a,b,c,f,N_last):
    #forward
    alpha=np.zeros(N_last)
    beta=np.zeros(N_last)
    alpha[0]=b[0]/c[0]
    beta[0]=f[0]/c[0]
    for i in range(1,N_last):
        alpha[i]=b[i]/(c[i]-a[i]*alpha[i-1])
        beta[i]=(f[i]+a[i]*beta[i-1])/(c[i]-a[i]*alpha[i-1])
    #backward
    sol=np.zeros(N_last+1)
    sol[N_last]=(f[N_last]+a[N_last]*beta[N_last-1])/(c[N_last]-a[N_last]*alpha[N_last-1])
    for i in range(N_last-1,-1,-1):
        sol[i]=alpha[i]*sol[i+1]+beta[i]
    return sol

#main
w=np.zeros((J+1,N+1,M+1))#numerical
w_an=np.zeros((J+1,N+1,M+1))#analytical
err=np.zeros((J+1,N+1,M+1))#error

w_int=np.zeros((N+1,M+1))#j+1/2 layer
x_line=np.zeros(N+1)#for storing results from trimatrix_solver
y_line=np.zeros(M+1)

#initial and boundary
#are already set to 0 by np


for j in range(J):
    #j -> j+1/2
    for m in range(1,M):
        #set matrix coefficients, find x_line, put into w_int
        c=np.zeros(N+1)
        b=np.zeros(N+1)###in this two 1 element is redundant
        a=np.zeros(N+1)###and kept for indexing
        f=np.zeros(N+1)
        c[0]=-1
        b[0]=0
        a[N]=0
        c[N]=-1
        f[0]=0
        f[N]=0
        for i in range(1,N):
            c[i]=tau/(h_x*h_x)+1
            a[i]=b[i]=tau/(2*h_x*h_x)
        for n in range(1,N):
            f[n]=w[j][n][m]+tau/(2*h_y*h_y)*(w[j][n][m+1]-2*w[j][n][m]+w[j][n][m-1])+0.5*tau*np.sin(n*h_x)*np.sin((j+0.5)*tau)
        x_line=trimatrix_solver(a,b,c,f,N)
        for n in range(1,N):
            w_int[n][m]=x_line[n]        
    #j+1/2 -> j+1
    for n in range(1,N):
        c=np.zeros(M+1)
        b=np.zeros(M+1)###in this two 1 element is redundant
        a=np.zeros(M+1)###and kept for indexing
        f=np.zeros(M+1)
        c[0]=-1
        b[0]=0
        a[M]=0
        c[M]=-1
        f[0]=0
        f[M]=0
        for i in range(1,M):
            c[i]=tau/(h_y*h_y)+1
            a[i]=b[i]=tau/(2*h_y*h_y)
        for m in range(1,M):
            f[m]=w_int[n][m]+tau/(2*h_x*h_x)*(w_int[n+1][m]-2*w_int[n][m]+w_int[n-1][m])+0.5*tau*np.sin(n*h_x)*np.sin((j+0.5)*tau)
        y_line=trimatrix_solver(a,b,c,f,M)
        for m in range(1,M):
            w[j+1][n][m]=y_line[m]

#fill values for analytical solution
S_ind=20
for j in range(J+1):
    for n in range(N+1):
        for m in range(M+1):
            sum=0
            for m_ind in range(1,S_ind+1):
                lam=1+(np.pi*m_ind/3)**2
                sum+=2*(1-(-1)**m_ind)/m_ind/np.pi*1/(1+lam**2)*(np.exp(-lam*j*tau)+lam*np.sin(j*tau)-np.cos(j*tau))*np.sin(n*h_x)*np.sin(np.pi*m_ind*m*h_y/3)
            w_an[j][n][m]=sum


#plot results
plt.figure(1)
axes=plt.axes(projection="3d")

x = np.linspace(0,l_x,N+1)#endpoint=true
y = np.linspace(0,l_y,M+1)

Xcoord_matr, Ycoord_matr = np.meshgrid(x, y, indexing='ij')
j=J
axes.plot_surface(Xcoord_matr, Ycoord_matr, w[j], cmap='Spectral_r')
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title("numerical, T="+str(T))
axes.view_init(elev=30, azim=-128)

plt.figure(2)
axes=plt.axes(projection="3d")

x = np.linspace(0,l_x,N+1)#endpoint=true
y = np.linspace(0,l_y,M+1)

Xcoord_matr, Ycoord_matr = np.meshgrid(x, y, indexing='ij')
j=J
axes.plot_surface(Xcoord_matr, Ycoord_matr, w_an[j], cmap='Spectral_r')
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title("analytical, T="+str(T))
axes.view_init(elev=30, azim=-128)

plt.figure(3)
axes=plt.axes(projection="3d")

x = np.linspace(0,l_x,N+1)#endpoint=true
y = np.linspace(0,l_y,M+1)

Xcoord_matr, Ycoord_matr = np.meshgrid(x, y, indexing='ij')
j=J
axes.plot_surface(Xcoord_matr, Ycoord_matr, w_an[j]-w[j], cmap='Spectral_r')
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title("analytical - numerical (error)")
axes.view_init(elev=30, azim=-128)

plt.show()