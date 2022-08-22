#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import numba as nb
import nu_nu_collisions as coll
import interp

from var_param import small_boxsize, eps_small_box, num_small_boxes, initial_boxsize


@nb.jit(nopython=True)
# Note: j is an index, not a p-value
def C_round(j,f1,p):
    c,c_frs = coll.cI(j,f1,p)
    if abs(c/c_frs) < 3e-15:
        return 0
    else:
        return c


@nb.jit(nopython=True)
def smallgrid(p,f,k0,k1):
    boxsize = p[-1] - p[-2]
    p_small_boxsize = p[1] - p[0]
#    print(int(1.25*k0), p[int(k1)-1], p_small_boxsize,len(p),k1)
    N = max(int(np.round(p[int(k1)-1]/p_small_boxsize,0))+1, int(1.25*k0))
    p_small = np.zeros(N)
    f_small = np.zeros(N)
    
    x_int = np.zeros(6)
    y_int = np.zeros(6)
    
    for i in range(num_small_boxes):
        p_small[i] = p[i]
        f_small[i] = f[i]
    for i in range(num_small_boxes,N):
        p_small[i] = i * p_small_boxsize
        k = int(np.round((p_small[i]-p[num_small_boxes-1])/boxsize,0)) + num_small_boxes - 1
        if np.round((p_small[i] - p[num_small_boxes-1]) / boxsize, 5) % 1 == 0:
#        if (p_small[i] - p[num_small_boxes-1]) % boxsize == 0:
            f_small[i] = f[k]
        else:
            if k+3 < len(p):
                for j in range(6):
                    x_int[j] = p[k + j - 2]
                    y_int[j] = f[k + j - 2]
                    if y_int[j] < 0:
                        print ("smallgrid",x_int[j], y_int[j])
                        print (k+j-2, k0, k1)
                        print ("f = ", f)
            else:
                x_int[:] = p[-6:]
                y_int[:] = f[-6:]
            f_small[i] = np.exp(interp.lin_int(p_small[i],x_int,np.log(y_int)))
    return p_small, f_small

@nb.jit(nopython=True)
def biggrid(p,f,k1):
    boxsize = p[-1] - p[-2]
    p_small_boxsize = p[1] - p[0]
    new_small_boxes = int(np.round(p[num_small_boxes-1]/boxsize,0)) + 1
    N = new_small_boxes + int(k1 - num_small_boxes)
    
    p_big = np.zeros(N)
    f_big = np.zeros(N)
    
    mult = int(round(boxsize/p_small_boxsize,0))
#    print(mult, new_small_boxes, N)
    p_big[0] = p[0]
    f_big[0] = f[0]
    for i in range(1,new_small_boxes):
        p_big[i] = p[mult * i]
        f_big[i] = np.max(f[mult*(i-1)+1:mult*i+1])
    for i in range(new_small_boxes, N):
        p_big[i] = p[num_small_boxes + (i-new_small_boxes)]
        f_big[i] = f[num_small_boxes + (i-new_small_boxes)]
        
    return p_big, f_big

#This only includes nu-nu, not nu-e;  nu-e added in model in driver()
@nb.jit(nopython=True,parallel=True)
def C_short(p,f1,T,k):
    c = np.zeros(len(p))
    boxsize = p[-1] - p[-2]

    if k[0] == 0:
        p_smallgrid, f_smallgrid = smallgrid(p,f1,num_small_boxes,len(p))
        p_wholegrid, f_wholegrid = biggrid(p,f1,len(p))
        for i in nb.prange(1,len(p)-1):
 #       for i in range(1,len(p)-1):
            if i < num_small_boxes:
                c[i] = C_round(i, f_smallgrid, p_smallgrid)
            else:
                c[i] = C_round((i-num_small_boxes)*2 + num_small_boxes , f_smallgrid, p_smallgrid)
 #               c[i] = C_round(i-num_small_boxes+1+int(eps_small_box/boxsize),f_wholegrid,p_wholegrid)
    else:
        k0 = num_small_boxes
#        print(k0, k[1], k[2], len(p))
        p_smallgrid, f_smallgrid = smallgrid(p,f1,k0,k[1])
        p_biggrid, f_biggrid = biggrid(p,f1,k[2])
        p_wholegrid, f_wholegrid = biggrid(p,f1,len(p))
#        print(p_biggrid)
#        for i in range(1,len(p)-1):
        for i in nb.prange(1,len(p)-1):
            if i < k0:
                c[i] = C_round(i, f_smallgrid, p_smallgrid)
    #            c[i] = C_round(i,f1[:k[1]],p[:k[1]])
    #            c[i] += ve.cI(i, f1[:k[1]],p[:k[1]],T)
            elif i < k[1]:
                c[i] = C_round(i-num_small_boxes+1+int(np.round(p[num_small_boxes-1]/boxsize,0)), f_biggrid, p_biggrid)
    #            c[i] = C_round(i,f1[:k[2]],p[:k[2]])
    #            c[i] += ve.cI(i, f1[:k[2]],p[:k[2]],T)
            else:
                c[i] = C_round(i-num_small_boxes+1+int(np.round(p[num_small_boxes-1]/boxsize,0)),f_wholegrid,p_wholegrid)
    #            c[i] += ve.cI(i,f1,p,T)
    return c




def simple_spread_eps(y,eps_array):
    old_boxsize = eps_array[num_small_boxes+1] - eps_array[num_small_boxes]
    new_boxsize = 2 * old_boxsize
    new_length = int((len(eps_array) - num_small_boxes)/2) + num_small_boxes
    y_new = np.zeros(new_length+2)
    eps_arr = np.zeros(new_length)
    y_new[-1] = y[-1]
    y_new[-2] = y[-2]
    y_new[0] = y[0]
    for i in range(1, num_small_boxes):
        eps_arr[i] = eps_array[i]
        y_new[i] = y[i]
    for i in range(num_small_boxes, new_length):
        eps_arr[i] = eps_arr[num_small_boxes-1] + (i+1 - num_small_boxes) * new_boxsize
        sec_trap_array=(y[num_small_boxes + 2*(i-num_small_boxes) + 1],y[num_small_boxes + 2*(i-num_small_boxes)])
        first_trap_array=(y[num_small_boxes + 2*(i-num_small_boxes)],y[num_small_boxes + 2*(i-num_small_boxes)-1])
        both_trap_array = (np.mean(first_trap_array),np.mean(sec_trap_array))
        y_new[i] = np.mean(both_trap_array)
    return y_new, eps_arr

def det_new_ics(a_i, T_i, f_SMALL, f_BUFFER, MIN_eps_BUFFER):
    a = a_i
    eps_small = - int(np.log(f_SMALL)/initial_boxsize)
    eps_buffer = max(eps_small + MIN_eps_BUFFER, - int(np.log(f_BUFFER)/initial_boxsize))

    initial_size = int((eps_buffer * initial_boxsize - eps_small_box)/initial_boxsize) + num_small_boxes
    y = np.zeros(initial_size + 3)
    eps_arr = np.zeros(initial_size)
    for i in range(num_small_boxes):
        eps_arr[i] = i * small_boxsize
        y[i] = 1 / (np.exp(eps_arr[i])+1)
    for i in range(num_small_boxes, len(eps_arr)):
        eps_arr[i] = eps_arr[num_small_boxes-1] + (i+1-num_small_boxes) * initial_boxsize
        y[i] = 1 / (np.exp(eps_arr[i])+1)
    y[-2] = T_i

    return a, y, eps_arr, eps_small, eps_buffer


# In[ ]:




