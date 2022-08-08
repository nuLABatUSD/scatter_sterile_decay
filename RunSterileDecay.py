import ODESolve
import numba as nb
import numpy as np
import derivatives as der
import interp
import coll_varbins as cv
import Collision_approx as ca
import os
import shutil

from Constants import mpi_neutral

from var_param import f_TINY, f_MINI, f_SMALL, f_BUFFER, MIN_eps_BUFFER, a_MAXMULT, eps_small_MAX, len_y_MAX, small_boxsize, eps_small_box, num_small_boxes, initial_boxsize


@nb.njit()
def infl(k):
    return int(np.round(k,0))

@nb.njit()
def find_breaks(f, E5_index = 0, E2_index = 0):
    k = np.ones(3) * (len(f) - 1)
    
    if (len(np.where(f < f_TINY)[0]) > 0):
        k[0] = np.where(f < f_TINY)[0][0]
    if (len(np.where(f < f_MINI)[0]) > 0):
        k[1] = np.where(f < f_MINI)[0][0]
    if (len(np.where(f < f_SMALL)[0]) > 0):
        k[2] = np.where(f < f_SMALL)[0][0]
    
    for i in range(infl(k[0]), len(f)):
        if f[i] > f_TINY:
            k[0] = i+1
    for i in range(infl(k[1]), len(f)):
        if f[i] > f_MINI:
            k[1] = i+1
    for i in range(infl(k[2]),len(f)):
        if f[i] > f_SMALL:
            k[2] = i+1

    for j in range(3):
        for i in [E5_index, E2_index]:
            if i - MIN_eps_BUFFER < k[j] <= i:
                k[j] += 2 * MIN_eps_BUFFER
            if i <= k[j] < i + MIN_eps_BUFFER:
                k[j] += MIN_eps_BUFFER
        for jj in range(j+1,3):
            if k[jj] < k[j] + MIN_eps_BUFFER:
                k[jj] = k[j] + MIN_eps_BUFFER
        if k[j] >= len(f):
            k[j] = len(f) - 1
    return k

@nb.njit()
def make_p(a0, eps, f, p_fix, A_model, n_model):
    len_eps = len(eps)
    p = np.zeros(len_eps + 11)
    p[:len_eps] = eps
    p[-4:] = p_fix
    p[-10] = len_eps
    p[-11] = eps[-1] - eps[-2]
    
    p[-8] = A_model
    p[-9] = n_model
    
    ms = p_fix[-1]
    EB5 = ms/2
    EB2 = (ms**2 - mpi_neutral**2)/(2*ms)
    
    p[-7:-4] = find_breaks(f, E5_index = np.where(eps < EB5 * a0)[0][-1], E2_index = np.where(eps < EB2 * a0)[0][-1])
    
    return p

@nb.njit()
def find_a_MAX(a0, t, p):
    ms = p[-1]
    tau = p[-2]
    D = p[-3]
    len_eps = infl(p[-10])
    deps_last = p[-11]
    
    EB5 = ms/2
    EB2 = (ms**2 - mpi_neutral**2)/(2*ms)
    
    
    f_FULL = 2 * np.pi**2 * der.nH(t, 1/a0, p[-2], p[-3]) / ( 1/a0 * EB5**2 * deps_last )
    decay_on = True
    
    if f_FULL < f_BUFFER:
        decay_on = False
        
    a_max = a0 * a_MAXMULT
    
    if decay_on:
        a_max = min(a_max, p[len_eps - 2 * MIN_eps_BUFFER - 1] / EB2)
    
    return a_max, decay_on

@nb.njit()
def driver(a0, eps0, y0, dx0, N_steps, dN, A_model, n_model, p_fix):
    a_MAX, decay_on = find_a_MAX(a0, y0[-1], make_p(a0, eps0, y0[:-2], p_fix, A_model, n_model))
    
    a = a0
    eps = np.copy(eps0)
    y = np.copy(y0)
    dx = dx0
    
    out_a = np.zeros(N_steps+1)
    out_y = np.zeros((N_steps+1, len(y0)))
    out_dx = np.zeros(N_steps+1)
    
    last_ind = N_steps
    
    eps_small = np.where(y0[:-2] > f_SMALL)[0][-1]
    eps_buffer = np.where(y0[:-2] > f_BUFFER)[0][-1]
    
    out_a[0] = a0
    out_y[0,:] = y0
    out_dx[0] = dx0
    
    for ns in range(N_steps):
        params = make_p(a, eps, y[:-2], p_fix, A_model, n_model)
                
        ode_out = ODESolve.ODEOneRun(a, y, dx, params, 1, dN, a_MAX)
        
        a = ode_out[0][-1]
        dx = ode_out[2][-1]
        y = np.copy(ode_out[1][-1])
        
        out_a[ns+1] = a
        out_y[ns+1][:] = y
        out_dx[ns+1] = dx
        
        if ode_out[3]:
            last_ind = ns + 1
            break
            
        eps_check = np.where(y[:-2] > f_SMALL)[0][-1]  
        if eps_check - eps_small > (eps_buffer - eps_small) / 2:
            last_ind = ns + 1
            break

            
    
    return out_a[:(last_ind+1)], out_y[:(last_ind+1),:], out_dx[:(last_ind+1)], decay_on

@nb.njit()
def forward(mH, y_v, e_array, a, decay_on):
    eps_small_new = np.where(y_v[:-2] > f_SMALL)[0][-1]
    xp = np.zeros(2)
    yp = np.zeros(2)
    kk = 0
    while (yp[1] >= yp[0]):
        if eps_small_new + kk >= len(e_array):
            break
        else:
            for i in range(2):
                yp[i] = y_v[eps_small_new + kk + i - 1]
                xp[i] = e_array[eps_small_new + kk + i - 1]
            kk += 1
            
    if eps_small_new + kk == len(e_array):
        print("f is increasing at last box?")
        return y_v, e_array
    
    eps_small_new = eps_small_new + kk
    bxsz = abs(xp[1] - xp[0])
    new_len = len(y_v)
    if y_v[eps_small_new] < f_BUFFER:
        new_len = eps_small_new
    else:
        if decay_on:
            e_up = np.where(e_array < 0.5 * mH * a)[0][-1]
            if e_up > len(e_array) - 2 * MIN_eps_BUFFER:
                e_up += MIN_eps_BUFFER * 3
            e_test = e_array[eps_small_new-1] + (new_len - (MIN_eps_BUFFER + 1) - eps_small_new) * bxsz
            while (e_test - 3*MIN_eps_BUFFER * bxsz)/a <= 0.5 * mH:
                e_up += MIN_eps_BUFFER
                e_test = e_array[eps_small_new-1] + (e_up - (MIN_eps_BUFFER + 1) - eps_small_new) * bxsz
            
            e_temp = max(eps_small_new, e_up)
        else:
            e_temp = eps_small_new
            
        if e_temp > len(y_v) - 5:
            new_len = e_temp
            
            eps_small_new = np.where(y_v[:-2] > np.sqrt(f_SMALL*f_BUFFER))[0][-1]
            yp[0] = y_v[eps_small_new-1]
            yp[1] = y_v[eps_small_new]
            xp[0] = e_array[eps_small_new-1]
            xp[1] = e_array[eps_small_new]
            
            while yp[1] > yp[0]:
                eps_small_new += 2
                yp[0] = y_v[eps_small_new-1]
                yp[1] = y_v[eps_small_new]
                xp[0] = e_array[eps_small_new-1]
                xp[1] = e_array[eps_small_new]
                
        else:
            if y_v[e_temp] < f_BUFFER:
                new_len = e_temp
                eps_small_new = e_temp
            else:
                new_len = e_temp
                
                yp[0] = y_v[e_temp]
                yp[1] = y_v[e_temp+1]
                xp[0] = e_array[e_temp]
                xp[1] = e_array[e_temp+1]
                e_extrap = e_temp

                while e_temp < len(y_v) - 5:
                    if y_v[e_temp] < f_BUFFER:
                        break
                    if y_v[e_temp+1] < y_v[e_temp] and yp[1]/yp[0] > y_v[e_temp+1]/y_v[e_temp]:
                        yp[0] = y_v[e_temp]
                        yp[1] = y_v[e_temp+1]
                        xp[0] = e_array[e_temp]
                        xp[1] = e_array[e_temp+1]
                        e_extrap = e_temp
                    e_temp += 2
                
                for i in range(20*eps_small_new):
                    if interp.log_linear_extrap(xp[0] + i * bxsz, xp, yp) > f_BUFFER:
                        new_len = i + e_extrap
                    else:
                        break
                    
                eps_small_new = e_extrap
    y = np.zeros(new_len+2)
    y[-1] = y_v[-1]
    y[-2] = y_v[-2]
    eps_array = np.zeros(new_len)
    for i in range(eps_small_new):
        eps_array[i] = e_array[i]
        y[i] = y_v[i]
    if len(eps_array) > eps_small_new:
        for i in range(eps_small_new, len(eps_array)):
            eps_array[i] = e_array[eps_small_new-1] + (i+1 - eps_small_new) * bxsz
            y[i] = interp.log_linear_extrap(eps_array[i], xp, yp)

    return y, eps_array

def nextstep(a0, eps0, y0, dx0, N_steps, dN, p_fix, fn, nr):
    
    input_filename = fn + "/inputs-{}".format(nr)
    output_filename = fn + "/full-{}".format(nr)
    
    A_model, n_model = ca.model_An(a0, y0[-2])

    np.savez(input_filename,ms=p_fix[-1],mixangle=p_fix[-4],a0=a0,y0=y0,dx0=dx0,e_array=eps0,A_model=A_model,n_model=n_model)
    
    result = driver(a0, eps0, y0, dx0, N_steps, dN, A_model, n_model, p_fix)
    
    np.savez(output_filename,ms=p_fix[-1],mixangle=p_fix[-4],a=result[0],y=result[1],dx=result[2],eps=eps0)
    
    y_next, eps_next = forward(p_fix[-1], result[1][-1], eps0, result[0][-1], result[3])
    
    return result[0][-1], eps_next, y_next, result[2][-1]


def run_code(a0, eps0, y0, dx0, p_fix, temp_fin, folder_name):
    a = a0
    eps = np.copy(eps0)
    y = np.copy(y0)
    dx = dx0
    for i in range(200):
        a, eps, y, dx = nextstep(a, eps, y, dx, 100, 100, p_fix, folder_name, i)
        
        eps_small = np.where(y[:-2] > f_SMALL)[0][-1]
        if eps_small > eps_small_MAX or len(y) > len_y_MAX:
            y, eps = cv.simple_spread_eps(y, eps)
            
        if y[-2] < temp_fin:
            break

def run_code_error(folder_name, inputs_file):    
    in_npz = np.load("{}/{}".format(folder_name,inputs_file))
    a = float(in_npz['a0'])
    y = in_npz['y0']
    eps = in_npz['e_array']
    dx = float(in_npz['dx0'])
    
    p_fix = make_p_fixed(float(in_npz['ms']), float(in_npz['mixangle']), 1/1.79**3)

    fn = "{}/debug".format(folder_name)    
    if not os.path.exists(fn):
        os.makedirs(fn)
    else:
        print("Folder {} already exists.  Overwriting any data".format(fn))
        
    shutil.copy("{}/{}".format(folder_name,inputs_file), "{}/debug/original-{}".format(folder_name, inputs_file))


    for i in range(200):
        a, eps, y, dx = nextstep(a, eps, y, dx, 10, 5, p_fix, fn, i)

def run_code_w_error(a0, eps0, y0, dx0, p_fix, temp_fin, folder_name):
    a = a0
    eps = np.copy(eps0)
    y = np.copy(y0)
    dx = dx0
    for i in range(200):
        try:
            a, eps, y, dx = nextstep(a, eps, y, dx, 100, 100, p_fix, folder_name, i)
            
            eps_small = np.where(y[:-2] > f_SMALL)[0][-1]
            if eps_small > eps_small_MAX or len(y) > len_y_MAX:
                y, eps = cv.simple_spread_eps(y, eps)
            
            if y[-2] < temp_fin:
                break
        except:
            try:
                run_code_error(folder_name, "inputs-{}.npz".format(i))
                print("Error occurred in Run {}.  Somehow error was not repeated".format(i))
            except:
                print("Error occurred in Run {}.  Debugging runs completed.".format(i))                
            return -1
    return i

@nb.njit()
def make_p_fixed(ms,mixangle,D):
    p_out = np.zeros(4,dtype=np.float64)
    p_out[-1] = ms
    p_out[-2] = der.tH(ms,mixangle)
    p_out[-3] = D
    p_out[-4] = mixangle
    return p_out

@nb.njit()
def det_new_ics(T_i):
    a = 1/T_i
    eps_small = - int(np.log(f_SMALL)/initial_boxsize)
    eps_buffer = max(eps_small + MIN_eps_BUFFER, - int(np.log(f_BUFFER)/initial_boxsize))

    initial_size = int((eps_buffer * initial_boxsize - eps_small_box)/initial_boxsize) + num_small_boxes
    y = np.zeros(initial_size + 2)
    eps_arr = np.zeros(initial_size)
    for i in range(num_small_boxes):
        eps_arr[i] = i * small_boxsize
        y[i] = 1 / (np.exp(eps_arr[i])+1)
    for i in range(num_small_boxes, len(eps_arr)):
        eps_arr[i] = eps_arr[num_small_boxes-1] + (i+1-num_small_boxes) * initial_boxsize
        y[i] = 1 / (np.exp(eps_arr[i])+1)
    y[-2] = T_i

    return a, y, eps_arr

def N_eff(a_f,T_f,f_final,e_a): #this code assumes that y_final does not have both neutrino & antineutrino information
    TCM = 1/a_f 
    e_dens = (TCM**4/(2*np.pi**2))*der.trapezoid(f_final*e_a**3,e_a)
    neff = e_dens/((7/4)*(4/11)**(4/3)*(np.pi**2/30)*(T_f)**4)
    return 6*neff #to account for all 3 flavors of neutrinos and all 3 flavors of antineutrinos



def run(ms, mixangle, T0 = 15, D = 1/1.79**3, Tfin = 0.001):
    p_fix = make_p_fixed(ms, mixangle, D)

    a0, y0, eps0 = det_new_ics(T0)
    dx0 = a0/1000

    print("Mass = {} MeV; Lifetime = {:.3} s".format(ms,p_fix[-2]*6.58e-22))
    
    fn = "{}-{:.4}".format(ms,mixangle)

    if not os.path.exists(fn):
        os.makedirs(fn)
    else:
        print("Folder {} already exists.  Overwriting any data".format(fn))

    i_final = run_code_w_error(a0, eps0, y0, dx0, p_fix, Tfin, fn)
    
    if i_final != -1:
        read_npz = np.load("{}/full-{}.npz".format(fn,i_final))
        af = read_npz['a'][-1]
        Tf = read_npz['y'][-1,-2]
        f_f = read_npz['y'][-1,:-2]
        e_f = read_npz['eps']
        print("N_eff = {:.5}".format(N_eff(af, Tf, f_f, e_f)))
    
    
#    print("N_eff = {}".format(run_Neff(fn+"full")))

