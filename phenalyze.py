import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from scipy.optimize import curve_fit
from scipy.special import erfcinv


from math import comb
from functools import reduce
import operator



###################################################################################################################################################
"""
Data visualization functions
"""

def plot_3ddata_all_h(data:pd.DataFrame, h_list:list[float], title="Figura sin título", view_init=[30,-60,0], fig_size=(7,7)):
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=view_init[0], azim=view_init[1], roll=view_init[2])
    for h_i in h_list:
        noise1 = data.loc[data.h==h_i]["Protein 1"]
        noise2 = data.loc[data.h==h_i]["Protein 2"]
        noise3 = data.loc[data.h==h_i]["Protein 3"]
        
        ax.scatter(noise1, noise2, noise3, label=fr"$h={h_i:.2}$")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlabel(r"$\eta_1^2$")
        ax.set_ylabel(r"$\eta_2^2$")
        ax.set_zlabel(r"$\eta_3^2$")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    # plt.close()



def plot_3ddata_each_h(data:pd.DataFrame, h_list:list[float], title="Figura sin título", view_init=[30,-60,0], fig_size=(7,7), show_axis=True):
    fig = plt.figure(figsize=fig_size)
    for i in range(len(h_list)):
        noise1 = data.loc[data.h==h_list[i]]["Protein 1"]
        noise2 = data.loc[data.h==h_list[i]]["Protein 2"]
        noise3 = data.loc[data.h==h_list[i]]["Protein 3"]

        ax = fig.add_subplot(2, len(h_list)//2, i+1, projection='3d')
        ax.scatter(noise1, noise2, noise3)
        ax.set_title(fr"$h={h_list[i]:.2}$")
        ax.set_xlabel(r"$\eta_1^2$")
        ax.set_ylabel(r"$\eta_2^2$")
        ax.set_zlabel(r"$\eta_3^2$")
        ax.set_xlim(min(data["Protein 1"]),max(data["Protein 1"]))
        ax.set_ylim(min(data["Protein 2"]),max(data["Protein 2"]))
        ax.set_zlim(min(data["Protein 3"]),max(data["Protein 3"]))
        ax.view_init(elev=view_init[0], azim=view_init[1], roll=view_init[2])

        if not show_axis:
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.get_zaxis().set_ticks([])

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    # plt.close()


###################################################################################################################################################
"""
Fitting functions
"""

def compute_3dfit(data:pd.DataFrame, h_list:list[float], model:callable, seed:list, lims:list, max_ef=1000000):
    params_list = []
    err_list = []
    R2_list = []
    for i in range(len(h_list)):
        noise1 = data.loc[data.h==h_list[i]]["Protein 1"]
        noise2 = data.loc[data.h==h_list[i]]["Protein 2"]
        noise3 = data.loc[data.h==h_list[i]]["Protein 3"]

        params, cov = curve_fit(f = model, xdata = [noise1, noise2], ydata = noise3, p0=seed[i], bounds=lims[i], maxfev=max_ef)

        err = np.sqrt(2)*erfcinv(2*(1-0.95))*np.sqrt(np.diag(cov))
        absError = model([noise1, noise2], *params)  - noise3
        # RMSE = np.sqrt(np.mean(np.square(absError)))
        Rsquared = 1.0 - (np.var(absError) / np.var(noise3))

        params_list.append(params)
        err_list.append(err)
        R2_list.append(Rsquared)

    return params_list, err_list, R2_list



def plot_3dfit_each_h(data:pd.DataFrame, h_list:list[float], model:callable, params_list:list, err_list:list, title="Figura sin título", view_init=[30,-60,0], fig_size=(7,7), show_axis=True):
    fig = plt.figure(figsize=fig_size)
    for i in range(len(h_list)):
        noise1 = data.loc[data.h==h_list[i]]["Protein 1"]
        noise2 = data.loc[data.h==h_list[i]]["Protein 2"]
        noise3 = data.loc[data.h==h_list[i]]["Protein 3"]

        ax = fig.add_subplot(2, len(h_list)//2, i+1, projection='3d')

        ax.scatter(noise1, noise2, noise3)
        
        # noise1_grid, noise2_grid = np.meshgrid(noise1, noise2)
        ax.plot_trisurf(noise1, noise2, model([noise1, noise2], *(params_list[i])), edgecolor='none')

        confidence = abs(100*err_list[i]/params_list[i])
        ax.set_title(fr"$h={h_list[i]:.2}: {list(confidence.astype(int))}$")

        ax.set_xlabel(r"$\eta_1^2$")
        ax.set_ylabel(r"$\eta_2^2$")
        ax.set_zlabel(r"$\eta_3^2$")
        ax.set_xlim(min(data["Protein 1"]),max(data["Protein 1"]))
        ax.set_ylim(min(data["Protein 2"]),max(data["Protein 2"]))
        ax.set_zlim(min(data["Protein 3"]),max(data["Protein 3"]))
        ax.view_init(elev=view_init[0], azim=view_init[1], roll=view_init[2])

        if not show_axis:
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.get_zaxis().set_ticks([])
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    # plt.close()



def plot_3dfit_one_h(data:pd.DataFrame, h_list:list[float], h_index:int, model:callable, params_list:list, err_list:list, title="Figura sin título", view_init=[30,-60,0], fig_size=(7,7)):
    fig = plt.figure(figsize=fig_size)

    noise1 = data.loc[data.h==h_list[h_index]]["Protein 1"]
    noise2 = data.loc[data.h==h_list[h_index]]["Protein 2"]
    noise3 = data.loc[data.h==h_list[h_index]]["Protein 3"]

    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(noise1, noise2, noise3)
    
    # noise1_grid, noise2_grid = np.meshgrid(noise1, noise2)
    ax.plot_trisurf(noise1, noise2, model([noise1, noise2], *(params_list[h_index])), edgecolor='none')

    confidence = abs(100*err_list[h_index]/params_list[h_index])
    ax.set_title(fr"$h={h_list[h_index]:.2}: {list(confidence.astype(int))}$")

    ax.set_xlabel(r"$\eta_1^2$")
    ax.set_ylabel(r"$\eta_2^2$")
    ax.set_zlabel(r"$\eta_3^2$")
    # ax.set_xlim(min(data["Protein 1"]),max(data["Protein 1"]))
    # ax.set_ylim(min(data["Protein 2"]),max(data["Protein 2"]))
    # ax.set_zlim(min(data["Protein 3"]),max(data["Protein 3"]))
    ax.view_init(elev=view_init[0], azim=view_init[1], roll=view_init[2])
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    # plt.close()


###################################################################################################################################################
"""
Residuals functions
"""

def plot_3dresiduals_all_h(data, h_list, model, params_list, title="Figura sin título", view_init=[30,-60,0], fig_size=(7,7)):
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=view_init[0], azim=view_init[1], roll=view_init[2])
    for i in range(len(h_list)):
        noise1 = data.loc[data.h==h_list[i]]["Protein 1"]
        noise2 = data.loc[data.h==h_list[i]]["Protein 2"]
        noise3 = data.loc[data.h==h_list[i]]["Protein 3"]
        
        ax.scatter(noise1, noise2, noise3-model([noise1, noise2], *(params_list[i])), label=fr"$h={h_list[i]:.2}$")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlabel(r"$\eta_1^2$")
        ax.set_ylabel(r"$\eta_2^2$")
        ax.set_zlabel(r"$\eta_3^2$")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    # plt.close()


###################################################################################################################################################
###################################################################################################################################################
# Two dimensions
###################################################################################################################################################
###################################################################################################################################################
"""
Data visualization functions
"""

def plot_2ddata_all_h(data:pd.DataFrame, h_list:list[float], title="Figura sin título", fig_size=(7,7)):
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    for h_i in h_list:
        noise1 = data.loc[data.h==h_i]["Protein 1"]
        noise2 = data.loc[data.h==h_i]["Protein 2"]
        
        ax.scatter(noise1, noise2, label=fr"$h={h_i:.2}$")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlabel(r"$\eta_1^2$")
        ax.set_ylabel(r"$\eta_2^2$")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    # plt.close()



def plot_2ddata_each_h(data:pd.DataFrame, h_list:list[float], title="Figura sin título", fig_size=(7,7), show_axis=True):
    fig = plt.figure(figsize=fig_size)
    for i in range(len(h_list)):
        noise1 = data.loc[data.h==h_list[i]]["Protein 1"]
        noise2 = data.loc[data.h==h_list[i]]["Protein 2"]

        ax = fig.add_subplot(2, len(h_list)//2, i+1)
        ax.scatter(noise1, noise2)
        ax.set_title(fr"$h={h_list[i]:.2}$")
        ax.set_xlabel(r"$\eta_1^2$")
        ax.set_ylabel(r"$\eta_2^2$")
        ax.set_xlim(min(data["Protein 1"]),max(data["Protein 1"]))
        ax.set_ylim(min(data["Protein 2"]),max(data["Protein 2"]))

        if not show_axis:
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    # plt.close()


##########################################################################################################################
"""
Fitting functions
"""

def compute_2dfit(data:pd.DataFrame, h_list:list[float], model:callable, seed:list, lims:list, max_ef=1000000):
    params_list = []
    err_list = []
    R2_list = []
    for i in range(len(h_list)):
        noise1 = data.loc[data.h==h_list[i]]["Protein 1"]
        noise2 = data.loc[data.h==h_list[i]]["Protein 2"]

        params, cov = curve_fit(f=model, xdata=noise1, ydata=noise2, p0=seed[i], bounds=lims[i], maxfev=max_ef)

        err = np.sqrt(2)*erfcinv(2*(1-0.95))*np.sqrt(np.diag(cov))
        absError = model(noise1, *params)  - noise2
        # RMSE = np.sqrt(np.mean(np.square(absError)))
        Rsquared = 1.0 - (np.var(absError) / np.var(noise2))

        params_list.append(params)
        err_list.append(err)
        R2_list.append(Rsquared)

    return params_list, err_list, R2_list



def plot_2dfit_each_h(data:pd.DataFrame, h_list:list[float], model:callable, params_list:list, err_list:list, title="Figura sin título", fig_size=(7,7), show_axis=True):
    fig = plt.figure(figsize=fig_size)
    for i in range(len(h_list)):
        noise1 = data.loc[data.h==h_list[i]]["Protein 1"]
        noise2 = data.loc[data.h==h_list[i]]["Protein 2"]

        ax = fig.add_subplot(2, len(h_list)//2, i+1)

        ax.scatter(noise1, noise2)
        ax.plot(noise1, model(noise1, *(params_list[i])))

        confidence = abs(100*err_list[i]/params_list[i])
        ax.set_title(fr"$h={h_list[i]:.2}: {list(confidence.astype(int))}$")

        ax.set_xlabel(r"$\eta_1^2$")
        ax.set_ylabel(r"$\eta_2^2$")
        ax.set_xlim(min(data["Protein 1"]),max(data["Protein 1"]))
        ax.set_ylim(min(data["Protein 2"]),max(data["Protein 2"]))

        if not show_axis:
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    # plt.close()



def plot_2dfit_one_h(data:pd.DataFrame, h_list:list[float], h_index:int, model:callable, params_list:list, err_list:list, title="Figura sin título", fig_size=(7,7)):
    fig = plt.figure(figsize=fig_size)

    noise1 = data.loc[data.h==h_list[h_index]]["Protein 1"]
    noise2 = data.loc[data.h==h_list[h_index]]["Protein 2"]

    ax = fig.add_subplot(111)

    ax.scatter(noise1, noise2)
    ax.plot(noise1, model(noise1, *(params_list[h_index])))

    confidence = abs(100*err_list[h_index]/params_list[h_index])
    ax.set_title(fr"$h={h_list[h_index]:.2}: {list(confidence.astype(int))}$")

    ax.set_xlabel(r"$\eta_1^2$")
    ax.set_ylabel(r"$\eta_2^2$")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    # plt.close()


##########################################################################################################################
"""
Residuals functions
"""

def plot_2dresiduals_all_h(data:pd.DataFrame, h_list:list[float], model:callable, params_list:list, title="Figura sin título", fig_size=(7,7)):
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    for i in range(len(h_list)):
        noise1 = data.loc[data.h==h_list[i]]["Protein 1"]
        noise2 = data.loc[data.h==h_list[i]]["Protein 2"]
        
        ax.scatter(noise1, noise2-model(noise1, *(params_list[i])), label=fr"$h={h_list[i]:.2}$")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlabel(r"$\eta_1^2$")
        ax.set_ylabel(r"$\eta_2^2$")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    # plt.close()


###################################################################################################################################################
###################################################################################################################################################
# Theoretical
###################################################################################################################################################
###################################################################################################################################################


η_g = 1.0
γ_z = (1/60 + 1/18)/2
k_z = γ_z/(η_g**2)

# k_r = (1 + 2)/2
γ_r1 = (1/8 + 1/3)/2

α_i = 0.0
# β_i = (2 + 3)/2
k_i = (1E3 + 1E4)/2
# h_i = 2.0
γ_ri = (1/8 + 1/3)/2

k_pi = (10 + 60)/2
γ = (1/60 + 1/18)/2


k_r_list = np.linspace(1,3,10)
b_list = np.linspace(1,3,10)
h_list = np.linspace(1,4,10)



def compute_ss(tt_params):
    kr, γr, kp, γp = tt_params
    r_ss = kr/γr
    p_ss = r_ss*kp/γp
    return  r_ss, p_ss


def compute_const_int_noise(tt_params):
    kr, γr, kp, γp = tt_params
    r_ss, p_ss = compute_ss(tt_params)
    return (1/p_ss) + (1/r_ss)*(γp/(γr+γp))


def compute_H_repressor(tt_params, hill_params):
    kr, γr, kp, γp = tt_params
    α, β, k, h = hill_params
    r_ss, p_ss = compute_ss(tt_params)
    return h*((1/(1+((k/p_ss)**h)))-(α/(α+(α+β)*((k/p_ss)**h))))


def compute_H_activator(tt_params, hill_params):
    kr, γr, kp, γp = tt_params
    α, β, k, h = hill_params
    r_ss, p_ss = compute_ss(tt_params)
    return h*((1/(1+((k/p_ss)**h)))-((α+β)/(α+β+(α*((k/p_ss)**h)))))

# def prod(iterable):
#     return reduce(operator.mul, iterable, 1)

def psi(n:int, i:int, k:int):
    return comb(2*n-i-k,n-i) * 2**(1+i+k-2*n)

def phi(n:int, i:int, k:int, activador:bool, tt_paramlist:list, hill_paramlist:list):
    Hs = np.ones(n-1)
    for i in range(n-1):
        if activador:
            Hs[i] = compute_H_activator()
        if not activador:
            Hs[i] = compute_H_repressor


    return (-1)**(k+i)


# def kappa(n:int, i:int, k:int):
#     return comb(2*(n-1)-i-k,n-i-1) * 2**(2+i+k-2*n)


# def compute_cosa_general


