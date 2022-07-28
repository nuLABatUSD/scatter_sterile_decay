#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
#import new_ve_Collisions1_interp_extrap_0210 as ve
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import numba as nb
import nu_e_collisions as ve
from Constants import Gf, me

# In[2]:


x_values, w_values = np.polynomial.laguerre.laggauss(100) 

#constants:
#Gf = 1.166*10**-11 #This is the fermi constant in units of MeV^-2
#me = .511          #Mass of an electron in MeV
g = 2           #mulitplicity, spin up and spin down


@nb.jit(nopython=True)
def f(p,Tcm,c): #occupation fraction for the neutrinos we've been using
    return 1/(np.e**(c*p/Tcm)+1)
@nb.jit(nopython=True)
def f_eq(p,T,eta): #occupation fraction of the neutrinos if they were in thermal equilibrium with the electrons & positrons
    return 1/(np.e**((p/T)-eta)+1)
@nb.jit(nopython=True)
def n_e(T): #number density of electrons
    Ep_array = np.sqrt(x_values**2 + me**2) #x_values are momenta here
    integral = np.sum((np.e**x_values)*w_values*(x_values**2)/(np.e**(Ep_array/T)+1))
    return (g/(2*np.pi**2))*integral 


# In[3]:


def cs_eta_old(T,Tcm,c):
    integrand = np.zeros(len(x_values))
    for i in range (len(x_values)):
        integrand[i] = (np.e**x_values[i])*w_values[i]*(x_values[i]**2)*f(x_values[i],Tcm,c)
    integral = np.sum(integrand) #value that will match with the eta we eventually output from this function
    
    Eta_array = np.linspace(-10,10,1000) #trying to make a range of etas that will encompass any potential eta
    integral_array = np.zeros(len(Eta_array)) #cubic spline integral that match w/ etas in the eta_array
    hold = np.zeros(len(x_values))
    for i in range (len(Eta_array)):
        for j in range (len(x_values)):
            hold[j] = (np.e**x_values[j])*w_values[j]*(x_values[j]**2)*f_eq(x_values[j],T,Eta_array[i])
        integral_array[i] = np.sum(hold) 
    
    cs = CubicSpline(integral_array,Eta_array) #cs actually will be different each time, depends on T
    eta = cs(integral)
    return eta

@nb.njit()
def cs_njit(T,Tcm,c):
    integrand = np.zeros(len(x_values))
    for i in range (len(x_values)):
        integrand[i] = (np.e**x_values[i])*w_values[i]*(x_values[i]**2)*f(x_values[i],Tcm,c)
    integral = np.sum(integrand) #value that will match with the eta we eventually output from this function
    
    Eta_array = np.linspace(-10,10,1000) #trying to make a range of etas that will encompass any potential eta
    integral_array = np.zeros(len(Eta_array)) #cubic spline integral that match w/ etas in the eta_array
    hold = np.zeros(len(x_values))
    for i in range (len(Eta_array)):
        for j in range (len(x_values)):
            hold[j] = (np.e**x_values[j])*w_values[j]*(x_values[j]**2)*f_eq(x_values[j],T,Eta_array[i])
        integral_array[i] = np.sum(hold) 

    return integral, integral_array, Eta_array

def cs_eta(T,Tcm,c):
    
    integral, integral_array, Eta_array = cs_njit(T,Tcm,c)
    
    cs = CubicSpline(integral_array,Eta_array) 
    eta = cs(integral)
    return eta
    
# In[4]:



def model_An_old(a,T,c,npts=201,etop=20): 
    
    e_array = np.linspace(0,etop,int(npts))
    boxsize = e_array[1]-e_array[0]
    eta = cs_eta(T,1/a,c)
    
    ne = n_e(T)
    
    p_array = e_array / a
    f_array = f(p_array,1/a,c)
    feq_array = f_eq(p_array,T,eta)
    net = ve.driver_short(p_array,T,f_array,boxsize*(1/a))

    ## do not njit this... takes longer with constant need to re-compile
    def C_local(p_array,A,n):
        C_array = np.zeros(len(p_array))
        for i in range (len(C_array)):
            C_array[i] = (p_array[i]**n)*(f_array[i]-feq_array[i])
        C_array = -A*ne*(Gf**2)*(T**(2-n))*C_array
        return C_array
    popt, popc = curve_fit(C_local,e_array[1:int(0.5*len(e_array))]*(1/a),net[1:int(0.5*len(e_array))])
    A,n = popt

    
    return A,n

@nb.njit()
def min_max_eta(eta_list,c_net,p_arr,f_arr,T0):
    match_zero = np.where(c_net<0)[0][-1]
    i_min = -1
    i_max = -1
    match = False
    for i in range(len(eta_list)):
        temp = f_arr - f_eq(p_arr,T0,eta_list[i])
        tt = np.where(temp>0)
        if len(tt[0]) != 0:
            if(tt[0][-1] >= match_zero):
                if tt[0][-1] == match_zero:
                    match = True
                if not match:
                    i_min = i
            else:
                i_max = i
                break

    return(eta_list[i_min], eta_list[i_max])

@nb.njit()
def fit_model_params(c_net,p_arr,f_arr,T0,min_n = 0.5):
    if n_e(T0) < 1e-20:
        return 0., 1., 0.
    eta_list1 = np.linspace(-10,10,201)
    min1, max1 = min_max_eta(eta_list1, c_net, p_arr, f_arr, T0)
    eta_list2 = np.linspace(min1, max1, 201)
    min2, max2 = min_max_eta(eta_list2, c_net, p_arr, f_arr, T0)

    eta_vals = np.linspace(min2,max2,5)
    n_arr = np.linspace(min_n,min_n+1,101)

    aa=np.where(c_net > 0.25*max(c_net))[0]
    bb=np.where(c_net < 0.25*min(c_net))[0]
    LSQ = np.zeros((len(eta_vals), len(n_arr), len(n_arr)))
    A_vals = np.zeros_like(LSQ)

    for i in range(len(eta_vals)):
        temp = - Gf**2 * (f_arr - f_eq(p_arr,T0,eta_vals[i])) * n_e(T0)

        for j in range(len(n_arr)):
            temp2 = temp * p_arr**n_arr[j] * T0**(2-n_arr[j])

            A0 = min(c_net) / min(temp2)
            A_vals[i,j,:] = np.linspace(0.5 * A0, 1.5 * A0, len(n_arr))

            for k in range(len(A_vals[i,j,:])):
                diffA = temp2[aa] * A_vals[i,j,k] - c_net[aa]
                diffB = temp2[bb] * A_vals[i,j,k] - c_net[bb]

                LSQ[i,j,k] = np.sum(diffA**2) + np.sum(diffB**2)
    bf = np.argmin(LSQ)
    etaf = int(bf/101/101)
    nf = int((bf - etaf*101*101)/101)
    Af = bf - etaf*101*101 - nf * 101

    eta_fit = eta_vals[etaf]
    n_fit = n_arr[nf]
    A_fit = A_vals[etaf, nf, Af]
    
    return eta_fit, n_fit, A_fit


@nb.njit()
def f_nu(p, T):
    return 1/(np.exp(p/T)+1)


@nb.njit()
def model_An(a0, T0):
    if n_e(T0) < 1e-20:
        return 0., 1.
    e_arr = np.linspace(0,20,201)
    boxsize = e_arr[1]-e_arr[0]
    p_arr = e_arr / a0
    f_arr = f_nu(p_arr,T0*0.9)
    net = ve.driver_short(p_arr,T0,f_arr,boxsize*(1/a0))
    
    eta_fit, n_fit, A_fit = fit_model_params(net[:100], p_arr[:100], f_arr[:100], T0)
    if n_fit == 1.5:
        eta_fit, n_fit, A_fit = fit_model_params(net[:100], p_arr[:100], f_arr[:100], T0, min_n=1.5)
    return A_fit, n_fit
