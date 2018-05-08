import numpy as np
import scipy.optimize as opt
from numba import jit
from collections import defaultdict
import sys

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# BIPARTITE JACOBIAN

@jit()
def diagonal_block(xx, denoms, beta, row, dim, costs):
    """
    Returns the diagonal matrices to be included in the Jacobian computation.
    Each row represents the derivative of equations i = 1...row and alpha = 1...col
    with respect to x_i_beta and x_alpha_beta, for the same beta/rating.
    (d k_i_beta/d x_i_beta for i = 1...row
     d k_i_beta/d x_alpha_beta for i = 1...row and alpha = 1...col
     d k_alpha_beta/d x_i_beta for i = 1...row and alpha = 1...col
     d k_alpha_beta/d x_alpha_beta for alpha = 1...col)
    :param xx: vector of Lagrange multipliers which have to be solved for
    :type xx: numpy.array
    :param denoms: denominator of each term of the equation
     denoms[i,alpha] = sum x_i_beta*x_alpha_beta, for beta = 1...max_beta
    :type denoms: numpy.array
    :param beta: rating level considered (diagonal block of the Jacobian matrix)
    :type beta: int
    :return: the considered block
    * costs * if you are solving the resuced system, costs is the vector stating how many times
    each degree appears. If you are solving the full system, just set costs equal to a dim * max_meta
    vector of ones. 
    """
    block_1 = np.zeros((dim, dim))
    x_i = xx[0 + beta*dim : row + beta*dim]
    x_alpha = xx[row + beta*dim : dim + beta*dim]
    cost_i = costs[0 + beta*dim : row + beta*dim]
    cost_a = costs[row + beta*dim : dim + beta*dim]
    for i in range(0, row):
        block_1[i,i] = (x_alpha * denoms[i] * cost_a - xx[i + beta*dim]*np.square(x_alpha * denoms[i])*cost_a).sum()
        block_1[i, row:dim] = xx[i + beta*dim]*(denoms[i] - xx[i + beta*dim]*(x_alpha*np.square(denoms[i])))*cost_a
    for j in range(row, dim):
        block_1[j,j] = (x_i * denoms[:,j-row] * cost_i - xx[j + beta*dim]*np.square(x_i * denoms[:,j-row])*cost_i).sum()
        block_1[j, 0:row] = xx[j + beta*dim]*(denoms[:,j-row] - xx[j + beta*dim]*(x_i*np.square(denoms[:,j-row])))*cost_i
    return block_1
    

@jit()
def other_blocks(xx, denoms, beta, beta_prime, row, dim, costs):
    """
    Returns the off-diagonal matrices to be included in the Jacobian computation.
    Each row represents the derivative of equations i = 1...row and alpha = 1...col
    with respect to x_i_beta' and x_alpha_beta', for different betas/ratings.
    (d k_i_beta/d x_i_beta' for i = 1...row and beta' = 1...max_beta
     d k_i_beta/d x_alpha_beta' for i = 1...row, alpha = 1...col and beta' = 1...max_beta
     d k_alpha_beta/d x_i_beta' for i = 1...row, alpha = 1...col and beta' = 1...max_beta
     d k_alpha_beta/d x_alpha_beta' for alpha = 1...col and beta' = 1...max_beta)
    :param xx: vector of Lagrange multipliers which have to be solved for
    :type xx: numpy.array
    :param denoms: denominator of each term of the equation
     denoms[i,alpha] = sum x_i_beta*x_alpha_beta, for beta = 1...max_beta
    :type denoms: numpy.array
    :param beta: rating level considered for the equation 
    :type beta: int
    :param beta_prime: rating level considered for derivative 
    :type beta_prime: int
    :return: the considered block
    * costs * if you are solving the resuced system, costs is the vector stating how many times
    each degree appears. If you are solving the full system, just set costs equal to a dim * max_meta
    vector of ones. 
    """
    block_2 = np.zeros((dim, dim))
    x_alpha = xx[row + beta*dim : dim + beta*dim]
    x_alpha_prime = xx[row + beta_prime*dim : dim + beta_prime*dim]
    x_i = xx[0 + beta*dim : row + beta*dim]
    second_x_i = xx[0 + beta_prime*dim : row + beta_prime*dim]
    cost_i = costs[0 + beta*dim : row + beta*dim]
    cost_a = costs[row + beta*dim : dim + beta*dim]
    for i in range(0, row):
        x_i_prime = xx[i + beta_prime*dim]
        block_2[i,i] = - xx[i + beta*dim]*(x_alpha * x_alpha_prime * np.square(denoms[i]) * cost_a).sum()
        block_2[i, row:dim] = - xx[i + beta*dim]*x_i_prime*((x_alpha * np.square(denoms[i]))) * cost_a
    for j in range(row, dim):
        second_x_alpha = xx[j + beta_prime*dim]
        block_2[j,j] = - xx[j + beta*dim]*(x_i*second_x_i * np.square(denoms[:,j-row]) * cost_i).sum()
        block_2[j, 0:row] = - xx[j + beta*dim]*second_x_alpha*((x_i * np.square(denoms[:,j-row]))) * cost_i
    return block_2
    

@jit()
def jacobian(xx, dseq, max_beta, row, dim, costs):
    """
    Returns the Jacobian matrix of the system of equations.
    :param xx: vector of Lagrange multipliers which have to be solved for
    :type xx: numpy.array
    :return: Jacobian
    :type Jacobian: numpy.array
    * costs * if you are solving the resuced system, costs is the vector stating how many times
    each degree appears. If you are solving the full system, just set costs equal to a dim * max_meta
    vector of ones. 
    """    
#    np.savetxt('current_xx.txt', xx, fmt = '%.18f')
    jack = np.zeros((max_beta*dim, max_beta*dim))
    denoms = np.zeros((row, dim-row))
    for i in range(0, row):
        for alpha in range(row, dim):
            zum = 0.
            for bb in range(max_beta):
                zum += xx[i + bb*dim]*xx[alpha + bb*dim]
            denoms[i, alpha-row] = 1./(1. + zum)
    for beta in range(0, max_beta):
        for beta_prime in range(beta, max_beta):
            if beta == beta_prime:
                jack[beta*dim:dim+beta*dim, beta*dim:dim+beta*dim] = diagonal_block(xx, denoms, beta, row, dim, costs)
            else:
                jack[beta*dim:dim+beta*dim, beta_prime*dim:dim+beta_prime*dim] = other_blocks(xx, denoms, beta, beta_prime, row, dim, costs)
                jack[beta_prime*dim:dim+beta_prime*dim, beta*dim:dim+beta*dim] = other_blocks(xx, denoms, beta_prime, beta, row, dim, costs)
    return jack


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# MONOPARTITE JACOBIAN

@jit()
def monopartite_diagonal_block(xx, denoms, beta, nodes, costs):
    block_1 = np.zeros((nodes, nodes))
    for i in range(0 + beta*nodes, nodes + beta*nodes):
        ix = i - beta*nodes
        zum = 0.
        for j in range(0 + beta*nodes, nodes + beta*nodes):
            jx = j - beta*nodes
            zum += xx[j]*costs[j]*(denoms[ix, jx] - xx[j]*xx[i]*(denoms[ix, jx])**2)
            block_1[ix, jx] = xx[i]*costs[j]*(denoms[ix, jx] - xx[i]*xx[j]*(denoms[ix, jx])**2)
        block_1[ix, ix] = zum
    return block_1

@jit()
def monopartite_other_block(xx, denoms, beta, beta_prime, nodes, costs):
    block_2 = np.zeros((nodes, nodes))
    for i in range(0 + beta*nodes, nodes + beta*nodes):
        ix = i - beta*nodes
        zum = 0.
        for j in range(0 + beta*nodes, nodes + beta*nodes):
            jx = j - beta*nodes
            zum += costs[j]*(xx[j]*xx[jx + beta_prime*nodes]*(denoms[ix, jx])**2)
            block_2[ix, jx] = - costs[j]*(xx[j]*xx[i]*xx[ix + beta_prime*nodes]*(denoms[ix, jx])**2)
        block_2[ix,ix] = - xx[i]*zum
    return block_2

@jit()
def monopartite_jacobian(xx, dseq, max_beta, nodes, costs):
    #np.savetxt('current_xx_wiki.txt', xx, fmt = '%.18f')
    jack = np.zeros((max_beta*nodes, max_beta*nodes))
    denoms = np.zeros((nodes, nodes))
    for i in range(0, nodes):
        for j in range(0, nodes):
            zum = 0.
            for bb in range(max_beta):
                zum += xx[i + bb*nodes]*xx[j + bb*nodes]
            denoms[i, j] = 1./(1. + zum)
    for beta in range(0, max_beta):
        for beta_prime in range(beta, max_beta):
            if beta == beta_prime:
                jack[beta*nodes:nodes+beta*nodes, beta*nodes:nodes+beta*nodes] = monopartite_diagonal_block(xx, denoms, beta, nodes, costs)
            else:
                jack[beta*nodes:nodes+beta*nodes, beta_prime*nodes:nodes+beta_prime*nodes] = monopartite_other_block(xx, denoms, beta, beta_prime, nodes, costs)
                jack[beta_prime*nodes:nodes+beta_prime*nodes, beta*nodes:nodes+beta*nodes] = monopartite_other_block(xx, denoms, beta_prime, beta, nodes, costs)
    return jack

#------------------------------------------------------------------------------------------------------------
# EQUATIONS

@jit()
def monopartite_equations(xx, dseq, max_beta, nodes, costs):
    #sys.stdout.flush()
    denoms = np.zeros((nodes, nodes))
    eq = - dseq
    sum_completed = 0
    for beta in range(max_beta):
        for i in range(0 + beta*nodes, nodes + beta*nodes):
            for j in range(i+1, nodes + beta*nodes):
                if sum_completed == 0:
                    zum = 0.
                    for bb in range(max_beta):
                        zum += xx[i + bb*nodes]*xx[j + bb*nodes]
                    denoms[i, j] = 1./(1. + zum)
                tt = xx[i]*xx[j]*denoms[i - beta*nodes, j - beta*nodes]
                eq[i] += tt * costs[j]
                eq[j] += tt * costs[i]
        sum_completed = 1
    return eq

@jit()
def equations(xx, dseq, max_beta, row, dim, costs):
    """
    Returns the system of equations to be solved.
    :param xx: vector of Lagrange multipliers which have to be solved for
    :type xx: numpy.array
    * costs * if you are solving the resuced system, costs is the vector stating how many times
    each degree appears. If you are solving the full system, just set costs equal to a dim * max_meta
    vector of ones. 
    """    
    #sys.stdout.flush()
    denoms = np.zeros((row, dim-row))
    eq = -dseq
    sum_completed = 0
    for beta in range(max_beta):
        for i in range(0 + beta*dim, row + beta*dim):
            for alpha in range(row + beta*dim, dim + beta*dim):
                if sum_completed == 0:
                    zum = 0.
                    for bb in range(max_beta):
                        zum += xx[i + bb*dim]*xx[alpha + bb*dim]
                    denoms[i, alpha-row] = 1./(1. + zum)
                tt = xx[i]*xx[alpha]*denoms[i - beta*dim, alpha - row - beta*dim]
                eq[i] += tt * costs[alpha]
                eq[alpha] += tt * costs[i]
        sum_completed = 1
    return eq

#----------------------------------------------------------------------------------------------------------
# GET MATRICES

@jit()
def get_matrix_mono(sol, n_nodes, max_beta):
    denoms = np.zeros((n_nodes, n_nodes))
    mat = np.zeros((n_nodes, int(max_beta)*n_nodes))
    sum_completed = 0
    for beta in range(max_beta):
        for i in range(0 + beta*n_nodes, n_nodes + beta*n_nodes):
            for j in range(i+1, n_nodes + beta*n_nodes):
                if sum_completed == 0:
                    zum = 0.
                    for bb in range(max_beta):
                        zum += sol[i + bb*n_nodes]*sol[j + bb*n_nodes]
                    denoms[i, j] = (1. + zum)
                tt = sol[i]*sol[j]/denoms[i - beta*n_nodes, j - beta*n_nodes]
                mat[i - beta*n_nodes, j] = tt
                mat[j - beta*n_nodes, i] = tt
        sum_completed = 1
    return mat

@jit()
def get_matrix(sol, row, dim, max_beta):
    """
    Computes a probability matrix of dimensions (row, max_beta*col).
    Each block of dimension (row, col) represents the probability matrix for each rating. 
    Blocks correspond to betas in decreasing order: 
      first block -> probability matrix for max_beta, 
      second block -> probability matrix for (max_beta - 1), 
      etc.
    :param sol: solutions of the system of equations (always full solution, if you have solved
                the reduced system, first transform the solution into the extended one.)
    :type sol: numpy.array
    :returns: probability matrix of the null model
    """
    denoms = np.zeros((row, dim - row))
    mat = np.zeros((row, int(max_beta)*(dim - row)))
    sum_completed = 0
    for beta in range(max_beta):
        for i in range(0 + beta*dim, row + beta*dim):
            for alpha in range(row + beta*dim, dim + beta*dim):
                if sum_completed == 0:
                    zum = 0.
                    for bb in range(max_beta):
                        zum += sol[i + bb*dim]*sol[alpha + bb*dim]
                    denoms[i, alpha - row] = (1. + zum)
                tt = sol[i]*sol[alpha]/denoms[i - beta*dim, alpha - row - beta*dim]
                mat[i - beta*dim, alpha - (beta + 1)*row] = tt
        sum_completed = 1
    return mat
   

# ******* How to use the equations in a separate file: **********

# import BiSCM as biscm

#    * costs * if you are solving the resuced system, costs is the vector stating how many times
#    each degree appears. If you are solving the full system, just set costs equal to a dim * max_meta
#    vector of ones. 

# * dseq*
# vector of length (max_beta * tot_dim) that collects degrees for each rating.
# first tot_dim are degree sequence for max_beta, second tot_dim are degrees for max_beta-1 etc
# (decreasing betas).

# tot_dim = number of nodes (rows + cols)
# rows = nodes on one layer
# cols = nodes on the other
# max_bb = max rating

# l_equations = lambda x: biscm.equations(x, dseq, max_bb, rows, tot_dim, cost_vector)
# l_jac = lambda x: biscm.jacobian(x, dseq, max_bb, rows, tot_dim, cost_vector)
# sols = opt.least_squares(fun = l_equations, x0 = x0_init_guess, jac = l_jac, tr_solver = 'lsmr', xtol = 1e-12, max_nfev = 1000, verbose = 2)

# opzioni di least_squares da cambiare a occhio a seconda di come va il solver

# Probability matrix:

# probs = biscm.get_matrix(sol.x, rows, tot_dim, max_bb)
    