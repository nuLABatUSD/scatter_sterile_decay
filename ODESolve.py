import numpy as np
from derivatives import f as f
import numba as nb
import time

# Cash Karp coefficients
a2 = 1/5
a3 = 3/10
a4 = 3/5
a5 = 1
a6 = 7/8

b21 = 1/5
b31 = 3/40
b41 = 3/10
b51 = -11/54
b61 = 1631/55296
b32 = 9/40
b42 = -9/10
b52 = 5/2
b62 = 175/512
b43 = 6/5
b53 = -70/27
b63 = 575/13824
b54 = 35/27
b64 = 44275/110592
b65 = 253/4096

c1 = 37/378
c2 = 0
c3 = 250/621
c4 = 125/594
c5 = 0
c6 = 512/1771

cstar1 = 2825/27648
cstar2 = 0
cstar3 = 18575/48384
cstar4 = 13525/55296
cstar5 = 277/14336
cstar6 = 1/4


@nb.jit(nopython=True)
def RKCash_Karp(x, y, dx, p): 
    k1 = dx * f(x, y, p)
    k2 = dx * f(x + a2*dx, y + b21*k1, p)
    k3 = dx * f(x + a3*dx, y + b31*k1 + b32*k2, p)
    k4 = dx * f(x + a4*dx, y + b41*k1 + b42*k2 +b43*k3, p)
    k5 = dx * f(x + a5*dx, y + b51*k1 + b52*k2 + b53*k3 + b54*k4, p)
    k6 = dx * f(x + a6*dx, y + b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5, p)
    
    y_5th = y + c1*k1 + c2*k2 + c3*k3 + c4*k4 + c5*k5 + c6*k6 
    y_4th = y + cstar1*k1 + cstar2*k2 + cstar3*k3 + cstar4*k4 + cstar5*k5 + cstar6*k6 
    x_stepped = x + dx
    
    return x_stepped, y_5th, y_4th

@nb.jit(nopython=True)
def RKCK_step(x, y, dx, p):
    dx_try = dx
    for i in range(10):
        x_next, y5, y4 = RKCash_Karp(x, y, dx_try, p)
        dx_try, done = step_accept(y, y5, y4, dx_try)
        if done:
            break
    if not done:
        print("ERROR:  10 iterations without acceptable step")
        #np.savez("error-report", x = x, y = y, dx = dx)
        
    return x_next, y5, dx_try
    
eps = 1e-8
TINY = 1e-40
Safety = 0.9

@nb.jit(nopython=True)
def step_accept(y, y5, y4, dx):
    delta1 = np.abs(y5-y4)
    delta0 = eps * (np.abs(y) + np.abs(y5-y)) + TINY
    
    dsm = np.max(delta1/delta0)
    if dsm == 0:
        dx_new = 5 * dx
        return dx_new, True
    elif dsm < 1:
        dx_new = Safety * dx * dsm**-0.2
        dx_new = min(5.0 * dx, dx_new)
        return dx_new, True
    else:
        dx_new = Safety * dx * dsm**-0.25
        return dx_new, False
        
    
    
@nb.jit(nopython=True)
def ODEOneRun(x0, y0, dx0, p, N_step, dN, x_final):
    x_values = np.zeros(N_step+1)
    y_results = np.zeros((N_step+1, len(y0)))
    dx_array = np.zeros(N_step+1)
    
    x_values[0] = x0
    y_results[0,:] = y0
    dx_array[0] = dx0
    
    x = x0
    y = np.copy(y0)
    dx = dx0
    
    last_ind = N_step
    done = False
    for i in range(1,N_step+1):
        for j in range(dN):
            if x + dx > x_final:
                dx = x_final - x
            x, y, dx = RKCK_step(x, y, dx, p)
            if x == x_final:
                last_ind = i
                done = True
                break
        x_values[i] = x
        y_results[i,:] = y
        dx_array[i] = dx
        
        if done:
            break
        
    return x_values[:(last_ind+1)], y_results[:(last_ind+1),:], dx_array[:(last_ind+1)], done

def ODESolve_n_Save(x0, y0, dx0, p, N_run, N_step, dN, x_final, filename = "result", verbose=False, verbose_freq = 1, verbose_output = -1):
    x = x0
    y = np.copy(y0)
    dx = dx0
    st = time.time()
    for i in range(N_run):
        x_arr, y_mat, dx_arr, end = ODEOneRun(x, y, dx, p, N_step, dN, x_final)
        fn = "{}-{}".format(filename,i)
        np.savez(fn, x = x_arr, y = y_mat, dx = dx_arr)
        x = x_arr[-1]
        y = y_mat[-1,:]
        dx = dx_arr[-1]
        if verbose:
            if (i+1)%verbose_freq == 0:
                pc_run = (i+1)/N_run*100
                pc_time = (x-x0)/(x_final-x0)*100
                if pc_time > pc_run:
                    pct_done = pc_time
                    pct_text = "(to end point) "
                else:
                    pct_done = pc_run
                    pct_text = "(of total runs)"
                
                if verbose_output == -1 or verbose_output >= len(y):
                    read_out = x
                elif verbose_output == -100:
                    read_out = 1/x
                else:
                    read_out = y[verbose_output]
                print("{:8.6}; {:.4}% {}, total elapsed time = {:.4} min".format(read_out, pct_done, pct_text, (time.time()-st)/60))
            
        if end:
            break
    return x, y, dx


