import pandas as pd
import numpy as np
import numba
import random

@numba.jit()
def remove_percentage_links(A,perc=0.1):
    A_hat = np.copy(A)
    L = np.count_nonzero(A)
    L_miss = max(round(L*perc),1)
    i,j = np.nonzero(A_hat)
    ix = np.random.choice(len(i), L_miss, replace=False)
    for l in range(L_miss):
        A_hat[i[ix][l]][j[ix[l]]]=0    
    return A_hat, L_miss

@numba.jit()
def sort_probabilities(P,A_hat):
    D = P-A_hat #the observed links will have negative probabilities
    df = pd.DataFrame(D)
    df = df.unstack().reset_index()
    df.columns=['alpha','i','P']
    df = df[['i','alpha','P']]
    df = df[df['P']>0]
    df.sort_values('P',ascending=False,inplace=True)
    return df

@numba.jit()
def fill_miss_links(DF,A_hat,L_miss,weight,cond_exp_W=None):
    
    R = A_hat.copy()
    l = 0
    while l<L_miss:
        row = DF.iloc[l]  
        i = int(row.i)
        alpha = int(row.alpha)
        l+=1
        if weight == 0:
            R[i][alpha]=1

        else:
            R[i][alpha] = cond_exp_W[i][alpha]
    return R
