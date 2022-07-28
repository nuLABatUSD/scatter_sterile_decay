#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sympy as sym
import numpy as np
import numba as nb
from Interpolate import interp as ip

Weinberg = .4910015
x_values, w_values = np.polynomial.laguerre.laggauss(50)
x_valuese, w_valuese = np.polynomial.legendre.leggauss(50)
me = 0.511 
inf = 6457.2 


# In[2]:

x, y, p1, E2, E3, q3, q2, GF, stw = sym.symbols('x,y,p1,E2,E3,q3,q2,GF,stw')

M_1prime = 2**5 * GF**2 * (2 * stw + 1)**2 * ( x**2 - 2 * stw / (2 * stw + 1) *me**2*x )
M_2prime = 2**7 * GF**2 * (stw)**2 * ( x**2 + (2*stw + 1)/(2*stw) * me**2*x )

M_1_1 = sym.integrate( M_1prime.subs(x, ((p1+E2)**2-me**2-y**2)/2), (y, p1+E2-E3-q3, p1+E2-E3+q3) )
M_11 = sym.lambdify((p1,E2,E3,q3),M_1_1.subs([(GF,1.166e-11),(stw,(np.sin(Weinberg))**2)]))
M11 = nb.jit(M_11,nopython=True)

M_1_2 = sym.integrate( M_1prime.subs(x, ((p1+E2)**2-me**2-y**2)/2), (y, p1-q2, p1+q2) )
M_12 = sym.lambdify((p1,E2,q2),M_1_2.subs([(GF,1.166e-11),(stw,(np.sin(Weinberg))**2)]))
M12 = nb.jit(M_12,nopython=True)

M_1_3 = sym.integrate( M_1prime.subs(x, ((p1+E2)**2-me**2-y**2)/2), (y, E3+q3-p1-E2, p1+q2) )
M_13 = sym.lambdify((p1,E2,q2,E3,q3),M_1_3.subs([(GF,1.166e-11),(stw,(np.sin(Weinberg))**2)]))
M13 = nb.jit(M_13,nopython=True)

M_1_4 = sym.integrate( M_1prime.subs(x, ((p1+E2)**2-me**2-y**2)/2), (y, q2-p1, p1+E2-E3+q3) )
M_14 = sym.lambdify((p1,E2,q2,E3,q3),M_1_4.subs([(GF,1.166e-11),(stw,(np.sin(Weinberg))**2)]))
M14 = nb.jit(M_14,nopython=True)

M_2_1 = sym.integrate( M_2prime.subs(x, (y**2 + me**2 - (p1-E3)**2)/2), (y, p1-E3+E2-q2, p1-E3+E2+q2) )
M_21 = sym.lambdify((p1,E2,E3,q2),M_2_1.subs([(GF,1.166e-11),(stw,(np.sin(Weinberg))**2)]))
M21 = nb.jit(M_21,nopython=True)

M_2_2 = sym.integrate( M_2prime.subs(x, (y**2 + me**2 - (p1-E3)**2)/2), (y, p1-q3, p1+q3) )
M_22 = sym.lambdify((p1,E3,q3),M_2_2.subs([(GF,1.166e-11),(stw,(np.sin(Weinberg))**2)]))
M22 = nb.jit(M_22,nopython=True)

M_2_3 = sym.integrate( M_2prime.subs(x, (y**2 + me**2 - (p1-E3)**2)/2), (y, E3-p1-E2+q2, p1+q3) )
M_23 = sym.lambdify((p1,E2,q2,E3,q3),M_2_3.subs([(GF,1.166e-11),(stw,(np.sin(Weinberg))**2)]))
M23 = nb.jit(M_23,nopython=True)

M_2_4 = sym.integrate( M_2prime.subs(x, (y**2 + me**2 - (p1-E3)**2)/2), (y, q3-p1, p1-E3+E2+q2) )
M_24 = sym.lambdify((p1,E2,q2,E3,q3),M_2_4.subs([(GF,1.166e-11),(stw,(np.sin(Weinberg))**2)])) 
M24 = nb.jit(M_24,nopython=True)


# In[3]:


@nb.jit(nopython=True)
def trapezoid(array,dx):
    total = np.sum(dx*(array[1:]+array[:-1])/2)
    return total

@nb.jit(nopython=True)
def fe(E,T):
    return 1/(np.e**(E/T)+1)

@nb.jit(nopython=True)
def make_q_array(E_arr):
    q2_arr = E_arr**2 - 0.511**2
    q_arr = np.sqrt(q2_arr)
    if ((0.51099999 <= E_arr[0]) & (E_arr[0] <= .511)):
        q2_arr[0] = 0
        q_arr[0] = 0
    for i in range(len(q2_arr)):
        if (abs(q2_arr[i]) < 1e-13):
            q_arr[i] = 0
        elif (q2_arr[i]  < -1e-13):
            print("Error with q_array",q2_arr[i])
            q_arr[i] = 0
    return q_arr

@nb.jit(nopython=True)
def f_first_last(f, p4_arr, bx):
    j = max(int(p4_arr[0]/bx),int(p4_arr[0]/bx+1e-9))+1
    k = max(int(p4_arr[-1]/bx),int(p4_arr[-1]/bx+1e-9))
    p_arr = np.arange(len(f))*bx
    
    if (j<len(f)):
        f_first = ip.interp_log(p4_arr[0], p_arr, f)
    else:
        f_first = ip.log_linear_extrap(p4_arr[0],np.array([(len(f)-2)*bx,(len(f)-1)*bx]),np.array([f[-2],f[-1]]))
    if (k<len(f)-1):
        f_last = ip.interp_log(p4_arr[-1], p_arr, f)
    else:
        f_last = ip.log_linear_extrap(p4_arr[-1],np.array([(len(f)-2)*bx,(len(f)-1)*bx]),np.array([f[-2],f[-1]]))
    return f_first, f_last, j, k


# In[4]:


@nb.jit(nopython=True)
def Blim(p1,E,q,T,f,bx,sub,sup,n):
    
    if (sub==1): #it so happens that for R_1, only n matters for B limits, not superscript
        if ((2*p1 + E + q)==0):
            E1lim = inf
        else:
            E1lim = (1/2)*(2*p1 + E + q + (me**2)/(2*p1 + E + q))
        if ((2*p1 + E - q)==0):
            E2trans = inf
        else:
            E2trans = (1/2)*(2*p1 + E - q + (me**2)/(2*p1 + E - q))
        E2lim = E2trans
        if (n==1):
            UL = E
            LL = me
        elif (n==2):
            UL = E2trans
            LL = E
        elif (n==3):
            UL = E1lim
            LL = E2trans
        elif (n==4):
            UL = E2trans
            LL = me
        elif (n==5):
            UL = E
            LL = E2trans
        elif (n==6):
            UL = E1lim
            LL = E
        elif (n==7):
            UL = E
            LL = E2lim
        else: #n=8
            UL = E1lim
            LL = E
    
    else: #sub=2
        if ((E - q - 2*p1)==0):
            E1lim = inf
        else:
            E1lim = (1/2)*(E - q - 2*p1 + (me**2)/(E - q - 2*p1))
        if (E + q - 2*p1)==0:
            E2trans = inf
        else:
            E2trans = (1/2)*(E + q - 2*p1 + (me**2)/(E + q - 2*p1))
        E2lim = E2trans
        if (n==1):
            UL = E
            LL = me
        elif (n==2):
            if ((sup==1) or (sup==2) or (sup==3)):
                UL = E2trans
                LL = E
            else: #sup=4
                UL = bx*len(f)
                LL = E
        elif (n==3):
            if ((sup==1) or (sup==2) or (sup==3)):
                UL = E1lim
                LL = E2trans
            else: #sup=4
                UL = E
                LL = me
        elif (n==4):
            if ((sup==1) or (sup==2)):
                UL = E2trans
                LL = me
            elif (sup==3):
                UL = E
                LL = me
            else: #sup=4
                UL = E2trans
                LL = E
        elif (n==5):
            if ((sup==1) or (sup==2)):
                UL = E
                LL = E2trans
            elif (sup==3):
                UL = E2trans
                LL = E
            else: #sup=4
                UL = bx*len(f)
                LL = E2trans
        elif (n==6):
            if ((sup==1) or (sup==2)):
                UL = E1lim
                LL = E
            elif (sup==3):
                UL = bx*len(f)
                LL = E2trans
            else: #sup=4
                UL = E2trans
                LL = me
        elif (n==7):
            if (sup==1):
                UL = E
                LL = E2lim
            elif ((sup==2) or (sup==3)):
                UL = E2trans
                LL = me
            else: #sup=4
                UL = E
                LL = E2trans
        elif (n==8):
            if (sup==1):
                UL = E1lim
                LL = E
            elif ((sup==2) or (sup==3)):
                UL = E
                LL = E2trans
            else: #sup=4 
                UL = bx*len(f)
                LL = E
        elif (n==9):
            if ((sup==1) or (sup==4)):
                UL = E
                LL = E2lim
            else: #sup=2,3
                UL = bx*len(f)
                LL = E
        elif (n==10):
            if ((sup==1) or (sup==4)):
                UL = bx*len(f)
                LL = E
            else: #sup=2,3
                UL = E
                LL = E2lim
        else: #n=11
            UL = bx*len(f)
            LL = E
        
    return UL, LL


@nb.jit(nopython=True)
def Alim(p1,sub,sup,n):
    
    if (sub==1):
        E1cut = me + (2*p1**2)/(me - 2*p1)
        E3cut = np.sqrt(p1**2 + me**2)
        if (sup==1):
            if ((n==1) or (n==2) or (n==3)):
                UL = E3cut
                LL = me
            elif ((n==4) or (n==5) or (n==6)):
                UL = E1cut
                LL = E3cut
            else: #n=7,8
                UL = inf
                LL = E1cut
        else: #sup=2
            if ((n==1) or (n==2) or (n==3)):
                UL = E3cut
                LL = me
            else: #n=4,5,6
                UL = inf
                LL = E3cut
    
    else: #sub=2
        E1cut = p1 + (me**2)/(4*p1) 
        E2cut = p1 + me*(p1+me)/(2*p1+me)
        E3cut = np.sqrt(p1**2 + me**2)
        if (sup==1):
            if ((n==1) or (n==2) or (n==3)):
                UL = E3cut
                LL = me
            elif ((n==4) or (n==5) or (n==6)):
                UL = E2cut
                LL = E3cut
            elif ((n==7) or (n==8)):
                UL = E1cut
                LL = E2cut
            else: #n=9,10
                UL = inf
                LL = E1cut
        elif (sup==2):
            if ((n==1) or (n==2) or (n==3)):
                UL = E3cut
                LL = me
            elif ((n==4) or (n==5) or (n==6)):
                UL = E1cut
                LL = E3cut
            elif ((n==7) or (n==8) or (n==9)):
                UL = E2cut
                LL = E1cut
            else: #n=10,11
                UL = inf
                LL = E2cut
        elif (sup==3):
            if ((n==1) or (n==2) or (n==3)):
                UL = E1cut
                LL = me
            elif ((n==4) or (n==5) or (n==6)):
                UL = E3cut
                LL = E1cut
            elif ((n==7) or (n==8) or (n==9)):
                UL = E2cut
                LL = E3cut
            else: #n=10,11
                UL = inf
                LL = E2cut
        else: #sup=4
            if ((n==1) or (n==2)):
                UL = E1cut
                LL = me
            elif ((n==3) or (n==4) or (n==5)):
                UL = E3cut
                LL = E1cut
            elif ((n==6) or (n==7) or (n==8)):
                UL = E2cut
                LL = E3cut
            else: #n=9,10
                UL = inf
                LL = E2cut
    
    return UL, LL

@nb.jit(nopython=True)
def M(p1,E_arr,q_arr,E_val,q_val,sub,sup,n):
    M_arr = np.zeros(len(E_arr))
    
    if (sub==1): #it so happens that for R_1, only n matters for M, not superscript
        if ((n==1) or (n==4)):
            for i in range (len(E_arr)):
                M_arr[i] = M11(p1,E_val,E_arr[i],q_arr[i])
        elif (n==2):
            for i in range (len(E_arr)):
                M_arr[i] = M12(p1,E_val,q_val)
        elif ((n==3) or (n==6) or (n==8)):
            for i in range (len(E_arr)):
                M_arr[i] = M13(p1,E_val,q_val,E_arr[i],q_arr[i])
        else: #n=5,7
            for i in range (len(E_arr)):
                M_arr[i] = M14(p1,E_val,q_val,E_arr[i],q_arr[i])
    else: #sub=2
        if (((sup == 1) and ((n==1) or (n==4))) or (((sup==2) or (sup==3)) and ((n==1) or (n==4) or (n==7))) or ((sup==4) and ((n==1) or (n==3) or (n==6)))):
            for i in range (len(E_arr)):
                M_arr[i] = M21(p1,E_arr[i],E_val,q_arr[i])
        elif (((sup==1) and (n==2)) or ((sup==2) and (n==2)) or ((sup==3) and ((n==2) or (n==5))) or ((sup==4) and ((n==2) or (n==4)))):
            for i in range (len(E_arr)):
                M_arr[i] = M22(p1,E_val,q_val)
        elif (((sup==1) and ((n==3) or (n==6) or (n==8) or (n==10))) or (((sup==2) or (sup==3)) and ((n==3) or (n==6) or (n==9) or (n==11))) or ((sup==4) and ((n==5) or (n==8) or (n==10)))):
            for i in range (len(E_arr)):
                M_arr[i] = M23(p1,E_arr[i],q_arr[i],E_val,q_val)
        else: # (((sup==1) and ((n==5 or (n==7) or (n==9))) or ((sup==2) and ((n==5) or (n==8) or (n==10))) or ((sup==3) and ((n==8) or (n==10))) or ((sup==4) and ((n==7) or (n==9))))
            for i in range (len(E_arr)):
                M_arr[i] = M24(p1,E_arr[i],q_arr[i],E_val,q_val)

    return M_arr


# In[5]:


@nb.jit(nopython=True)
def B(p1,E_val,T,f,bx,sub,sup,n): #E_val can be either E3 or E2 depending on if it is R_1 or R_2
     
    q_val = (E_val**2 - .511**2)**(1/2)
    UL, LL = Blim(p1,E_val,q_val,T,f,bx,sub,sup,n)
    if (UL<LL):
        return 0,0
    p1_box = int(np.round(p1/bx,0))
    
    if (sub==1):
        UI = max(int((p1+E_val-LL)/bx),int((p1+E_val-LL)/bx+1e-9)) 
        LI = max(int((p1+E_val-UL)/bx),int((p1+E_val-UL)/bx+1e-9))
        if (UI<0 or LI<0):
            return 0,0
        len_p4 = min(UI - LI + 2, 2*len(f))
        p4_arr = np.zeros(len_p4)
        Fp_arr = np.zeros(len(p4_arr))
        Fm_arr = np.zeros(len(p4_arr))
        for i in range(len_p4-2):
            p4_arr[i+1] = (LI+i+1)*bx 
        p4_arr[0] = p1 + E_val - UL 
        p4_arr[-1] = p1 + E_val - LL
        E_arr = E_val + p1 - p4_arr
        q_arr = make_q_array(E_arr)
        M_arr = M(p1,E_arr,q_arr,E_val,q_val,sub,sup,n)
        for i in range(len_p4-2):
            if (LI+i+1 >= len(f)):
                f_holder = ip.log_linear_extrap(p4_arr[i+1],np.array([(len(f)-2)*bx, (len(f)-1)*bx]),np.array([f[-2],f[-1]]))
                if (f_holder < 1e-45):
                    break
            else:
                f_holder = f[LI+i+1] 
            Fp_arr[i+1] = (1-f[p1_box])*(1-fe(E_val,T))*fe(E_arr[i+1],T)*f_holder
            Fm_arr[i+1] = f[p1_box]*fe(E_val,T)*(1-fe(E_arr[i+1],T))*(1-f_holder)
        f_first, f_last, j, k = f_first_last(f, p4_arr, bx)
        Fp_arr[0] = (1-f[p1_box])*(1-fe(E_val,T))*fe(E_arr[0],T)*f_first
        Fm_arr[0] = f[p1_box]*fe(E_val,T)*(1-fe(E_arr[0],T))*(1-f_first)
        Fp_arr[-1] = (1-f[p1_box])*(1-fe(E_val,T))*fe(E_arr[-1],T)*f_last
        Fm_arr[-1] = f[p1_box]*fe(E_val,T)*(1-fe(E_arr[-1],T))*(1-f_last)
        
    else: #sub==2
        UI = max(int((p1+UL-E_val)/bx),int((p1+UL-E_val)/bx+1e-9))
        LI = max(int((p1+LL-E_val)/bx),int((p1+LL-E_val)/bx+1e-9))
        if (UI<0 or LI<0):
            return 0,0
        len_p4 = min(UI - LI + 2, 2*len(f))
        p4_arr = np.zeros(len_p4)
        Fp_arr = np.zeros(len(p4_arr))
        Fm_arr = np.zeros(len(p4_arr))
        for i in range(len_p4-2):
            p4_arr[i+1] = (LI+i+1)*bx
        p4_arr[0] = p1 + LL - E_val
        p4_arr[-1] = p1 + UL - E_val
        E_arr = E_val + p4_arr - p1
        q_arr = make_q_array(E_arr)
        M_arr = M(p1,E_arr,q_arr,E_val,q_val,sub,sup,n)
    
        for i in range(len_p4-2):
            if (LI+i+1 >=len(f)):
                f_holder = ip.log_linear_extrap(p4_arr[i+1],np.array([(len(f)-2)*bx,(len(f)-1)*bx]),np.array([f[-2],f[-1]]))
                if (f_holder < 1e-45):
                    break
            else:
                f_holder = f[LI+i+1] 
            Fp_arr[i+1] = (1-f[p1_box])*(1-fe(E_arr[i+1],T))*fe(E_val,T)*f_holder
            Fm_arr[i+1] = f[p1_box]*fe(E_arr[i+1],T)*(1-fe(E_val,T))*(1-f_holder)
    
        f_first, f_last, j, k = f_first_last(f, p4_arr, bx)
        Fp_arr[0] = (1-f[p1_box])*(1-fe(E_arr[0],T))*fe(E_val,T)*f_first
        Fm_arr[0] = f[p1_box]*fe(E_arr[0],T)*(1-fe(E_val,T))*(1-f_first)
        Fp_arr[-1] = (1-f[p1_box])*(1-fe(E_arr[-1],T))*fe(E_val,T)*f_last
        Fm_arr[-1] = f[p1_box]*fe(E_arr[-1],T)*(1-fe(E_val,T))*(1-f_last)
    
    igrndp_arr = Fp_arr*M_arr
    igrndm_arr = Fm_arr*M_arr
    igrlp = trapezoid(igrndp_arr[1:-1],bx)
    igrlm = trapezoid(igrndm_arr[1:-1],bx)
    igrlp = igrlp + 0.5*(igrndp_arr[0]+igrndp_arr[1])*(bx*j - p4_arr[0]) + 0.5*(igrndp_arr[-2]+igrndp_arr[-1])*(p4_arr[-1] - bx*k)
    igrlm = igrlm + 0.5*(igrndm_arr[0]+igrndm_arr[1])*(bx*j - p4_arr[0]) + 0.5*(igrndm_arr[-2]+igrndm_arr[-1])*(p4_arr[-1] - bx*k)
    return igrlp, igrlm

@nb.jit(nopython=True)
def A(p1,T,f,bx,sub,sup,n): 
    igrlp = 0.0
    igrlm = 0.0
    UL, LL = Alim(p1,sub,sup,n)
    if (UL == inf):
        E_arr = x_values+LL #could be E3 or E2 depending on if its R_1 or R_2
        for i in range (len(E_arr)):
            Bp, Bm = B(p1,E_arr[i],T,f,bx,sub,sup,n)
            igrlp = igrlp + (np.e**x_values[i])*w_values[i]*Bp
            igrlm = igrlm + (np.e**x_values[i])*w_values[i]*Bm     
    else:
        E_arr = ((UL-LL)/2)*x_valuese + (UL+LL)/2 #could be E3 or E2 depending on if its R_1 or R_2
        for i in range(len(E_arr)):
            Bp, Bm = B(p1,E_arr[i],T,f,bx,sub,sup,n)
            igrlp = igrlp + w_valuese[i]*((UL-LL)/2)*Bp
            igrlm = igrlm + w_valuese[i]*((UL-LL)/2)*Bm
    return np.array([igrlp, igrlm])


# In[6]:


@nb.jit(nopython=True)
def R11R21(p1,T,f,bx):
    coefficient = 1/(2**4*(2*np.pi)**3*p1**2)
    
    A11_1 = A(p1,T,f,bx,1,1,1)
    A11_2 = A(p1,T,f,bx,1,1,2)
    A11_3 = A(p1,T,f,bx,1,1,3)
    A11_4 = A(p1,T,f,bx,1,1,4)
    A11_5 = A(p1,T,f,bx,1,1,5)
    A11_6 = A(p1,T,f,bx,1,1,6)
    A11_7 = A(p1,T,f,bx,1,1,7)
    A11_8 = A(p1,T,f,bx,1,1,8)
    
    A21_1 = A(p1,T,f,bx,2,1,1)
    A21_2 = A(p1,T,f,bx,2,1,2)
    A21_3 = A(p1,T,f,bx,2,1,3)
    A21_4 = A(p1,T,f,bx,2,1,4)
    A21_5 = A(p1,T,f,bx,2,1,5)
    A21_6 = A(p1,T,f,bx,2,1,6)
    A21_7 = A(p1,T,f,bx,2,1,7)
    A21_8 = A(p1,T,f,bx,2,1,8)
    A21_9 = A(p1,T,f,bx,2,1,9)
    A21_10 = A(p1,T,f,bx,2,1,10)
    
    integral11 = A11_1 + A11_2 + A11_3 + A11_4 + A11_5 + A11_6 + A11_7 + A11_8 
    integral21 = A21_1 + A21_2 + A21_3 + A21_4 + A21_5 + A21_6 + A21_7 + A21_8 + A21_9 + A21_10
    integral = integral11 + integral21
    
    if (integral[0]==0 and integral[1]==0):
        return 0
    elif (abs((integral[0]-integral[1])/(integral[0]+integral[1]))<10**-14):
        return 0
    else:
        net = coefficient*(integral[0]-integral[1])
        return net

@nb.jit(nopython=True)
def R11R22(p1,T,f,bx):
    coefficient = 1/(2**4*(2*np.pi)**3*p1**2)
    
    A11_1 = A(p1,T,f,bx,1,1,1)
    A11_2 = A(p1,T,f,bx,1,1,2)
    A11_3 = A(p1,T,f,bx,1,1,3)
    A11_4 = A(p1,T,f,bx,1,1,4)
    A11_5 = A(p1,T,f,bx,1,1,5)
    A11_6 = A(p1,T,f,bx,1,1,6)
    A11_7 = A(p1,T,f,bx,1,1,7)
    A11_8 = A(p1,T,f,bx,1,1,8)
    
    A22_1 = A(p1,T,f,bx,2,2,1)
    A22_2 = A(p1,T,f,bx,2,2,2)
    A22_3 = A(p1,T,f,bx,2,2,3)
    A22_4 = A(p1,T,f,bx,2,2,4)
    A22_5 = A(p1,T,f,bx,2,2,5)
    A22_6 = A(p1,T,f,bx,2,2,6)
    A22_7 = A(p1,T,f,bx,2,2,7)
    A22_8 = A(p1,T,f,bx,2,2,8)
    A22_9 = A(p1,T,f,bx,2,2,9)
    A22_10 = A(p1,T,f,bx,2,2,10)
    A22_11 = A(p1,T,f,bx,2,2,11)
    
    integral11 = A11_1 + A11_2 + A11_3 + A11_4 + A11_5 + A11_6 + A11_7 + A11_8 
    integral22 = A22_1 + A22_2 + A22_3 + A22_4 + A22_5 + A22_6 + A22_7 + A22_8 + A22_9 + A22_10 + A22_11
    integral = integral11 + integral22
    
    if (integral[0]==0 and integral[1]==0):
        return 0
    elif (abs((integral[0]-integral[1])/(integral[0]+integral[1]))<10**-14):
        return 0
    else:
        net = coefficient*(integral[0]-integral[1])
        return net

@nb.jit(nopython=True)
def R11R23(p1,T,f,bx):
    coefficient = 1/(2**4*(2*np.pi)**3*p1**2)
    
    A11_1 = A(p1,T,f,bx,1,1,1)
    A11_2 = A(p1,T,f,bx,1,1,2)
    A11_3 = A(p1,T,f,bx,1,1,3)
    A11_4 = A(p1,T,f,bx,1,1,4)
    A11_5 = A(p1,T,f,bx,1,1,5)
    A11_6 = A(p1,T,f,bx,1,1,6)
    A11_7 = A(p1,T,f,bx,1,1,7)
    A11_8 = A(p1,T,f,bx,1,1,8)
    
    A23_1 = A(p1,T,f,bx,2,3,1)
    A23_2 = A(p1,T,f,bx,2,3,2)
    A23_3 = A(p1,T,f,bx,2,3,3)
    A23_4 = A(p1,T,f,bx,2,3,4)
    A23_5 = A(p1,T,f,bx,2,3,5)
    A23_6 = A(p1,T,f,bx,2,3,6)
    A23_7 = A(p1,T,f,bx,2,3,7)
    A23_8 = A(p1,T,f,bx,2,3,8)
    A23_9 = A(p1,T,f,bx,2,3,9)
    A23_10 = A(p1,T,f,bx,2,3,10)
    A23_11 = A(p1,T,f,bx,2,3,11)
    
    integral11 = A11_1 + A11_2 + A11_3 + A11_4 + A11_5 + A11_6 + A11_7 + A11_8 
    integral23 = A23_1 + A23_2 + A23_3 + A23_4 + A23_5 + A23_6 + A23_7 + A23_8 + A23_9 + A23_10 + A23_11
    integral = integral11 + integral23
    
    if (integral[0]==0 and integral[1]==0):
        return 0
    elif (abs((integral[0]-integral[1])/(integral[0]+integral[1]))<10**-14):
        return 0
    else:
        net = coefficient*(integral[0]-integral[1])
        return net

@nb.jit(nopython=True)
def R12R24(p1,T,f,bx):
    coefficient = 1/(2**4*(2*np.pi)**3*p1**2)
    
    A12_1 = A(p1,T,f,bx,1,2,1)
    A12_2 = A(p1,T,f,bx,1,2,2)
    A12_3 = A(p1,T,f,bx,1,2,3)
    A12_4 = A(p1,T,f,bx,1,2,4)
    A12_5 = A(p1,T,f,bx,1,2,5)
    A12_6 = A(p1,T,f,bx,1,2,6)
    
    A24_1 = A(p1,T,f,bx,2,4,1)
    A24_2 = A(p1,T,f,bx,2,4,2)
    A24_3 = A(p1,T,f,bx,2,4,3)
    A24_4 = A(p1,T,f,bx,2,4,4)
    A24_5 = A(p1,T,f,bx,2,4,5)
    A24_6 = A(p1,T,f,bx,2,4,6)
    A24_7 = A(p1,T,f,bx,2,4,7)
    A24_8 = A(p1,T,f,bx,2,4,8)
    A24_9 = A(p1,T,f,bx,2,4,9)
    A24_10 = A(p1,T,f,bx,2,4,10)

    integral12 = A12_1 + A12_2 + A12_3 + A12_4 + A12_5 + A12_6
    integral24 = A24_1 + A24_2 + A24_3 + A24_4 + A24_5 + A24_6 + A24_7 + A24_8 + A24_9 + A24_10
    integral = integral12 + integral24
    
    if (integral[0]==0 and integral[1]==0):
        return 0
    elif (abs((integral[0]-integral[1])/(integral[0]+integral[1]))<10**-14):
        return 0
    else:
        net = coefficient*(integral[0]-integral[1])
        return net

@nb.jit(nopython=True, parallel=True)
def driver(p_arr,T,f,bx):
    bx = p_arr[1]-p_arr[0] #why do we do this immediately if we send boxsize as an argument?
    output_arr = np.zeros(len(p_arr))
    for i in nb.prange (1,len(p_arr)):
        if (p_arr[i]<.15791):
            output_arr[i] = R11R21(p_arr[i],T,f,bx)
        elif (p_arr[i]<.18067):
            output_arr[i] = R11R22(p_arr[i],T,f,bx)
        elif (p_arr[i]<.2555):
            output_arr[i] = R11R23(p_arr[i],T,f,bx)
        else:
            output_arr[i] = R12R24(p_arr[i],T,f,bx)
    return output_arr

@nb.jit(nopython=True, parallel=True)
def driver_short(p_arr,T,f,bx):
    bx = p_arr[1]-p_arr[0] #why do we do this immediately if we send boxsize as an argument?
    output_arr = np.zeros(len(p_arr))
    for i in nb.prange (1,len(p_arr)//2):
        if (p_arr[i]<.15791):
            output_arr[i] = R11R21(p_arr[i],T,f,bx)
        elif (p_arr[i]<.18067):
            output_arr[i] = R11R22(p_arr[i],T,f,bx)
        elif (p_arr[i]<.2555):
            output_arr[i] = R11R23(p_arr[i],T,f,bx)
        else:
            output_arr[i] = R12R24(p_arr[i],T,f,bx)
    return output_arr

