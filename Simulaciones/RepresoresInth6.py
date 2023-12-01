import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit
import os

df_params = pd.read_csv(os.path.join(os.getcwd(),'params.csv'))
params_vals = list(df_params['value'])
γ_r1, α_i, β_i, k_i, γ_ri, k_pi, γ, kr_min, kr_max, b_min, b_max, h_min, h_max, η_g, γ_z = np.float_(params_vals)

# γ_r1 = (1/8 + 1/3)/2

# α_i = 0.0
# β_i = (2 + 3)/2
# k_i = (1E3 + 1E4)/2
# γ_ri = (1/8 + 1/3)/2

# k_pi = (10 + 60)/2
# γ = (1/60 + 1/18)/2


k_r_list = np.linspace(kr_min,kr_max,10)
β_list = np.linspace(b_min,b_max,10)
h_list = np.linspace(h_min,h_max,10)


@njit("f8(f8,f8,f8,f8,f8)")
def repressor(p, α, β, k, h):
    return α + β * (k**h / (k**h + p**h))

@njit("f8[:](f8[:],f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8)")
def propensity(state, k_r1, γ_r1, k_p1, γ_p1, α_2, β_2, k_2, h_2, γ_r2, k_p2, γ_p2, α_3, β_3, k_3, h_3, γ_r3, k_p3, γ_p3):
    r1, p1, r2, p2, r3, p3 = state

    s_1 = k_r1
    s_2 = γ_r1*r1
    s_3 = k_p1*r1
    s_4 = γ_p1*p1

    s_5 = repressor(p1, α_2, β_2, k_2, h_2)
    s_6 = γ_r2*r2
    s_7 = k_p2*r2
    s_8 = γ_p2*p2

    s_9 = repressor(p2, α_3, β_3, k_3, h_3)
    s_10 = γ_r3*r3
    s_11 = k_p3*r3
    s_12 = γ_p3*p3

    return np.array([s_1, s_2, s_3, s_4, s_5, s_6, s_7, s_8, s_9, s_10, s_11, s_12])



R = np.array([[1, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0], 
            [0, -1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0], 
            [0, 0, -1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0], 
            [0, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, -1]])


@njit("f8[:](f8[:],f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8)")
def evolve(state, k_r1, γ_r1, k_p1, γ_p1, α_2, β_2, k_2, h_2, γ_r2, k_p2, γ_p2, α_3, β_3, k_3, h_3, γ_r3, k_p3, γ_p3):
    prop = propensity(state, k_r1, γ_r1, k_p1, γ_p1, α_2, β_2, k_2, h_2, γ_r2, k_p2, γ_p2, α_3, β_3, k_3, h_3, γ_r3, k_p3, γ_p3)
    cum_prop = np.cumsum(prop/np.sum(prop))
    m = np.random.uniform(0,1)  
    i = np.searchsorted(cum_prop, m, side='right')

    new_state = state + R[i]

    return new_state



@njit("f8[:,:](f8[:],f8[:],f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8)")
def cell(T, state, k_r1, γ_r1, k_p1, γ_p1, α_2, β_2, k_2, h_2, γ_r2, k_p2, γ_p2, α_3, β_3, k_3, h_3, γ_r3, k_p3, γ_p3):
    n_samples = len(T)
    evolution = np.zeros((n_samples, len(state)))

    t = T[0]
    for i in range(len(T)):
        while t < T[i]:
            prop = propensity(state, k_r1, γ_r1, k_p1, γ_p1, α_2, β_2, k_2, h_2, γ_r2, k_p2, γ_p2, α_3, β_3, k_3, h_3, γ_r3, k_p3, γ_p3)
            dt = np.random.exponential(1/np.sum(prop))

            state = evolve(state, k_r1, γ_r1, k_p1, γ_p1, α_2, β_2, k_2, h_2, γ_r2, k_p2, γ_p2, α_3, β_3, k_3, h_3, γ_r3, k_p3, γ_p3)
            t+=dt
        evolution[i] = state
        
    return evolution


# @njit("f8[:,:,:](f8[:],i8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8)")
def culture(T, n_celulas, k_r1, γ_r1, k_p1, γ_p1, α_2, β_2, k_2, h_2, γ_r2, k_p2, γ_p2, α_3, β_3, k_3, h_3, γ_r3, k_p3, γ_p3):
    r1_ss = k_r1/γ_r1
    p1_ss = r1_ss*k_p1/γ_p1
    r2_ss = repressor(p1_ss, α_2, β_2, k_2, h_2)/γ_r2
    p2_ss = r2_ss*k_p2/γ_p2
    r3_ss = repressor(p2_ss, α_3, β_3, k_3, h_3)/γ_r3
    p3_ss = r3_ss*k_p3/γ_p3

    state = np.array([r1_ss, p1_ss, r2_ss, p2_ss, r3_ss, p3_ss]).round()
    # state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).round()

    results = np.zeros((n_celulas, len(T), len(state)))
    for i in range(n_celulas):
        trial = cell(T, state, k_r1, γ_r1, k_p1, γ_p1, α_2, β_2, k_2, h_2, γ_r2, k_p2, γ_p2, α_3, β_3, k_3, h_3, γ_r3, k_p3, γ_p3)
        results[i] = trial
    
    return results



def save_stats(Data, T, h_i, k_r1, β_2):
    ii = len(T)//5
    
    Data_mask = Data[:, ii:, :]
    RNA1_ss, protein1_ss, RNA2_ss, protein2_ss, RNA3_ss, protein3_ss = Data_mask.T
    RNA1_ss.ravel() 
    protein1_ss.ravel()
    RNA2_ss.ravel()
    protein2_ss.ravel()
    RNA3_ss.ravel()
    protein3_ss.ravel()
    
    Matrix = [[RNA1_ss.mean(), protein1_ss.mean(), RNA2_ss.mean(), protein2_ss.mean(), RNA3_ss.mean(), protein3_ss.mean()],
            [pow(RNA1_ss.std(),2)/pow(RNA1_ss.mean(),2), pow(protein1_ss.std(),2)/pow(protein1_ss.mean(),2), pow(RNA2_ss.std(),2)/pow(RNA2_ss.mean(),2), pow(protein2_ss.std(),2)/pow(protein2_ss.mean(),2), pow(RNA3_ss.std(),2)/pow(RNA3_ss.mean(),2), pow(protein3_ss.std(),2)/pow(protein3_ss.mean(),2)]]

    df = pd.DataFrame(np.array(Matrix), columns=["RNA 1", "Protein 1", "RNA 2", "Protein 2", "RNA 3", "Protein 3"], index=["Mean", "CV2"])

    df.to_csv(f"RepInt_h{h_i:.1f}_kr{k_r1:.1f}_b{β_2:.1f}.csv")

    return df



t_f = 300.0
T = np.arange(0.0, t_f, 1.0)
n_celulas = 10000


h_i = h_list[6]

for k_r in k_r_list:
    for β_2 in tqdm(β_list):
        params = [k_r, γ_r1, k_pi, γ,   α_i, β_2, k_i, h_i, γ_ri, k_pi, γ,   α_i, β_i, k_i, h_i, γ_ri, k_pi, γ]

        Data = culture(T, n_celulas, *params)
        save_stats(Data, T, h_i, k_r, β_2)