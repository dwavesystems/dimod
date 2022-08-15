# -*- coding: utf-8 -*-
# Copyright 2022 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# ===============================================================================================

#Author: Jack Raymond
#Date: December 18th 2020

import numpy as np
import dimod 

from typing import Callable, Optional, Sequence, Union, Iterable

def cdma(num_var: int = 64,  var_per_unit_bandwidth: float = 1.5, SNR: float = 5, *,
         discreteSS: bool = True, random_state: Optional[Union[np.random.RandomState, int]] = None,
         noise_discretization: float = None, constellation: str = 'BPSK',
         planted_state: Iterable = None) -> tuple:
    """Generate a cdma/mimo ensemble problem over a Gaussian channel.
 
    A multi-user channel problem for which the maximum likelihood problem 
    is a QUBO/Ising-optimization problem. A more effective optimizer allows use of 
    lower bandwidth (cost), whilst maintaining decoding quality.

    Channel model for a vector of transmitted complex valued symbols v.
    vec(signal) = spreading_sequence/sqrt(num_var) vec(v) + sigma vec(nu) ; nu = nu_R + i nu_I ; nu ~ N(0,1/2) (Gaussian channel)
    The special case of CDMA, with binary binary phase shift keying is further 
    discussed (code is under development to support other cases):
    Inference problem (maximum likelihood == Minimization of Hamiltonian)
    H(s) = -||signal - ss/sqrt(num_var) v||^2/(2 sigma^2) = H0 + s' J s + h s
    In recovery mode s=1 is the ground state and H(s) ~ -N/2;
    1. Transmitted state is all 1, which is the unique ground state in low noise limit. And with high probability the ground state up to some critical noise threshold. 
    2. Defaults: SNR = 7dB = 10^0.7 = 5, var_per_unit_bandwidth = 1.5, discreteSS=true, is a near critical regime where at scale N=64, all 1 is the ground state with probability ~ 50%. Reducing noise (or increasing bandwidth), transmission is successful with high probability (given an ideal optimizer), if reduced significantly (increased) optimization becomes easy (for most heuristics). By contrast increasing noise (decreasing bandwidth), decoding fails to recover the transmitted sequence with high probability (even with ideal optimizer).  
    3. Be sure to apply a spin-reversal transmormation for solvers that 
    are not spin-reversal invariant (e.g. QPU in particular)
    Args:
        num_var: number of variables (number of channel users, equiv. bits tranmitted)
        var_per_unit_bandwidth: num_var/bandwidth, cleanest (in practice constrained) to 
           choose such that num_var and bandwidth are integer.
        SNR: signal to noise ratio for the Gaussian channel.
        discreteSS: set to true for binary (+1,-1) valued spreading sequences, set to false for Gaussian 
           spreading sequences. Only expert users should change the default.
           discreteSS is actually a misnomer (BPSK applies regardless of the spreading sequence
           or channel).
        random_state: a numpy pseudo random number generator, or a seed thereof
        noise_discretization: We can discretize the noise ensemble such that the problem 
           is integer valued. At fixed finite SNR, and as either N or noise discretization
           (or both) becomes large, we recover the standard ensemble up to a prefactor. 
           Be careful at large SNR. Note that although J is even integer, the typical value 
           scales as sqrt(N)*noise_discretization. h is integer and the typical value scales
           as sqrt(N)*noise_discretization/SNR. The largest absolute values follows Gumbel 
           distributions (larger by approximately ~log(N)). 
           After discretization |J| values are 0,2|noise_discretization|,4|noise_discretization|,..
           After discretization |h| values are p,p+2,p+4,.. etc. p = 1 or 2.
    Returns:
        Tuple: First element is the binary quadratic model, other things of less interest.
        
    .. [#] T. Tanaka IEEE TRANSACTIONS ON INFORMATION THEORY, VOL. 48, NO. 11, NOVEMBER 2002
    .. [#] J. Raymond, N. Ndiaye, G. Rayaprolu and A. D. King, "Improving performance of logical qubits by parameter tuning and topology compensation," 2020 IEEE International Conference on Quantum Computing and Engineering (QCE), Denver, CO, USA, 2020, pp. 295-305, doi: 10.1109/QCE49297.2020.00044.
    .. [#] 
    .. Various D-Wave internal documentation
    """
    random_state = np.random.RandomState(random_state)
    assert num_var > 0, "Expect channel users"
    assert SNR > 0, "Expect positive signal to noise ratio"
    sigma = 1/np.sqrt(2*SNR);
    bandwidth = int(num_var/var_per_unit_bandwidth + random_state.random(1))
    assert bandwidth > 0, "Expect positive bandwidth (var_per_unit_bandwidth too large, or num_var too small)"

    if constellation == 'BPSK':
        num_spins = num_var
    elif constellation == 'QPSK':
        #Transmission & detection in both real and imaginary basis
        num_spins = 2*num_var
    elif constellation == '16QAM':
        num_spins = 4*num_var
    else:
        raise ValueError('Unknown constellation')
    #Real part of the channel:
    if discreteSS:
        spreading_sequence = (1-2*random_state.randint(2,size=(bandwidth,num_var)));
    else:
        assert noise_discretization == None, "noise_discretization not supported"
        spreading_sequence = random_state.normal(0,1,(bandwidth,num_var));
        
    spreading_sequence_scale = np.sqrt(num_var)
    
    white_gaussian_noise = spreading_sequence_scale*random_state.normal(0,sigma,(bandwidth,1))

    if constellation != 'BPSK':
        #Need imaginary part
        if discreteSS:
            spreading_sequenceI = (1-2*random_state.randint(2,size=(bandwidth,num_var)));
        else:
            spreading_sequenceI = random_state.normal(0,1,(bandwidth,num_var));
        white_gaussian_noiseI = spreading_sequence_scale*random_state.normal(0,sigma,(bandwidth,1))
    
    #Create integer valued Hamiltonian, noise precision is discretized relative to scale (1)
    #of the spreading_sequence (already in minimal integer form):
    if noise_discretization:
        assert float(noise_discretization).is_integer(), "scaling should be integer valued"
        assert noise_discretization>0, "scaling should be positive"
        spreading_sequence = spreading_sequence*noise_discretization
        #Naive discretization for now. Playing some extra tricks is possible, based on use in summation.
        #See also https://stackoverflow.com/questions/37411633/how-to-generate-a-random-normal-distribution-of-integers
        white_gaussian_noise = np.round(white_gaussian_noise*noise_discretization)
        spreading_sequence_scale = spreading_sequence_scale*noise_discretization
        if constellation != 'BPSK':
            spreading_sequenceI = spreading_sequence*noise_discretization
            white_gaussian_noiseI = np.round(white_gaussian_noise*noise_discretization)
    # Real part:
    # y = W 1 +  sqrt(N) nu
    if planted_state is None:
        if constellation != '16QAM':
            transmitted_symbols = 1
            transmitted_symbolsI = 1
        else:
            #By default, need to use random values:
            transmitted_symbols = random_state.choice([-3,-1,1,3],size=(1,num_var))
            transmitted_symbolsI = random_state.choice([-3,-1,1,3],size=(1,num_var))
    else:
        if constellation == 'BPSK' and len(planted_state) != num_var:
            raise ValueError('planted state is wrong length, should be iterable of num_var real values')
        else:
            if len(planted_state) != 2*num_var:
                raise ValueError('planted state is wrong length, should be iterable of 2*num_var real values')
            transmitted_symbolsI = np.array([[planted_state[i] for i in range(num_var, 2*num_var)]], dtype=float)
        transmitted_symbols = np.array([[planted_state[i] for i in range(num_var)]], dtype=float)
    # BPSK (real-real) part
    signal = np.reshape(np.sum(spreading_sequence*transmitted_symbols, axis=1),(bandwidth,1)) + white_gaussian_noise
    E0 = sum(signal*signal)
    J = np.matmul(spreading_sequence.T,spreading_sequence)
    h = - 2*np.matmul(spreading_sequence.T,signal)
    if constellation != 'BPSK':
        # See https://confluence.dwavesys.com/display/~jraymond/QPSK+and+16QAM+MIMO
        # [Real Mixed; Mixed Imag]
        signal += np.reshape(np.sum(spreading_sequenceI*transmitted_symbolsI, axis=1),(bandwidth,1))
        signalI = np.reshape(np.sum(spreading_sequence*transmitted_symbolsI, axis=1),(bandwidth,1)) \
                  + np.reshape(np.sum(spreading_sequenceI*transmitted_symbols, axis=1),(bandwidth,1)) \
                  + white_gaussian_noiseI;
        E0 += sum(signalI*signalI)
        h += - 2*np.matmul(spreading_sequenceI.T,signalI)
        h = np.concatenate((h, - 2*np.matmul(spreading_sequence.T,signalI) - 2*np.matmul(spreading_sequenceI.T,signal)),axis=0)
        J = np.concatenate((np.concatenate((J, np.matmul(spreading_sequence.T,spreading_sequenceI)), axis=0),
                           np.concatenate((np.matmul(spreading_sequenceI.T,spreading_sequence), np.matmul(spreading_sequenceI.T,spreading_sequenceI)), axis=0)),
                           axis=1)
        if constellation == '16QAM':
            # Outer product under linear encoding:
            h = np.kron(h, np.array([[1],[2]],dtype=float))
            J = np.kron(J, np.array([[1,2],[2,4]],dtype=float))
           
           
        
    natural_scale = (spreading_sequence_scale*spreading_sequence_scale)*SNR
    if noise_discretization == None:
        h = h/natural_scale
        J = J/natural_scale
        E0 = E0/natural_scale
    else:
        #Integers are quadratic in noise_discretization level, and have a prefactor 4
        h = h/(2*noise_discretization)
        J = J/(2*noise_discretization)
        natural_scale = natural_scale/(2*noise_discretization)
        E0 = E0/(2*noise_discretization) #Transmitted signal energy
    
    couplingDict = {}
    hDict = {}
    for u in range(num_var):
        hDict[u] = h[u]
        for v in range(u+1,num_var):
            couplingDict[(u,v)] = J[u][v] + J[v][u]
    bqm = dimod.BinaryQuadraticModel(hDict,couplingDict,dimod.Vartype.SPIN)
    return bqm, random_state, natural_scale, E0
