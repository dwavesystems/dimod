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

from typing import Callable, Sequence, Union, Iterable

def _quadratic_form(y,F):
    '''Convert O(v) = ||y - F v||^2 to a sparse quadratic form, where
    y,F are assumed to be complex or real valued.

    Constructs coefficients for the form O(v) = v^dag J v - 2 Re [h^dag vD] + k
    
    Inputs
        v: column vector of complex values
        y: column vector of complex values
        F: matrix of complex values
    Output
        k: real scalar
        h: dense real vector
        J: dense real symmetric matrix
    
    '''
    if len(y.shape) != 2 or y.shape[1] != 1:
        raise ValueError('y should have shape [n,1] for some n')
    if len(F.shape) != 2 or F.shape[0] != y.shape[0]:
        raise ValueError('F should have shape [n,m] for some m,n'
                         'and n should equal y.shape[1]')

    offset = np.matmul(y.imag.T,y.imag) + np.matmul(y.real.T,y.real)
    h = - 2*np.matmul(F.T.conj(),y) ## Be careful with interpretaion!
    J = np.matmul(F.T.conj(),F) 

    return offset,h,J

def _real_quadratic_form(h,J):
    '''Unwraps objective function on complex variables onto objective
    function of concatenated real variables: the real and imaginary
    parts.
    '''
    if h.dtype == np.complex128 or J.dtype == np.complex128:
        h = np.concatenate((h.real,h.imag),axis=0)
        J = np.concatenate((np.concatenate((J.real,J.imag),axis=0),
                            np.concatenate((J.imag.T,J.real),axis=0)),
                           axis=1)
    return h,J

def _amplitude_modulated_quadratic_form(h,J,modulation):
    if modulation == 'BPSK' or modulation == 'QPSK':
        #Easy case, just extract diagonal
        pass
    else:
        #Quadrature + amplitude modulation
        if modulation == '16QAM':
            num_amps = 2
        elif modulation == '64QAM':
            num_amps = 3
        else:
            raise ValueError('unknown modulation')
        amps = 2**np.arange(num_amps)
        h = np.kron(h,amps[:,np.newaxis])
        J = np.kron(J,np.kron(amps[:,np.newaxis],amps[np.newaxis,:]))
    
    return h, J

def spin_encoded_mimo(modulation: str, y: np.array = None, F: np.array = None,  
                      *,
                      num_var: int = None,  bandwidth: int = None, SNR: float = float('Inf'),
                      seed: Union[None, int, np.random.RandomState] = None,
                      transmitted_symbols: Iterable = None,
                      F_distribution: str = 'Normal',
                      use_offset: bool = False):
    """ Generate a multi-input multiple-output (MIMO) channel-decoding problem.
        
    Users each transmit complex valued symbols over a random channel :math:`F` of 
    some bandwidth, subject to additive white Gaussian noise. Given the received
    signal y the log likelihood of a given symbol set :math:`v` is given by 
    :math:`MLE = argmin || y - F v ||_2`. When v is encoded as a linear
    sum of spins the optimization problem is defined by a Binary Quadratic Model. 
    Depending on arguments used, this may be a model for Code Division Multiple
    Access _[#T02,#R20], 5G communication network problems _[#Prince], or others.
    
    Args:
        y: A complex or real valued signal in the form of a numpy array. If not
            provided, generated from other arguments.

        F: A complex or real valued channel in the form of a numpy array. If not
            provided, generated from other arguments.

        modulation: Specifies the constellation (symbol set) in use by 
            each user. Symbols are assumed to be transmitted with equal probability.
            Options are:
               * 'BPSK'
                   Binary Phase Shift Keying. Transmitted symbols are +1,-1;
                   no encoding is required.
                   A real valued channel is assumed.

               * 'QPSK'
                   Quadrature Phase Shift Keying. 
                   Transmitted symbols are +1,-1, +1j, -1j;
                   spins are encoded as a real vector concatenated with an imaginary vector.
                   
               * '16QAM'
                   Each user is assumed to select independently from 16 symbols.
                   The transmitted symbol is a complex value that can be encoded by two spins
                   in the imaginary part, and two spins in the real part. v = 2 s_1 + s_2.
                   Highest precision real and imaginary spin vectors, are concatenated to 
                   lower precision spin vectors.
                   
               * '64QAM'
                   A QPSK symbol set is generated, symbols are further amplitude modulated 
                   by an independently and uniformly distributed random amount from [1,3].

        num_var: Number of transmitted symbols, must be consistent with F.

        bandwidth: Bandwidth of channel.

        SNR: Signal to noise ratio. When y is not provided, this is used to 
            generate the noisy signal. In the case float('Inf') no noise is 
            added.

        
        transmitted_symbols: 
            The set of symbols transmitted, this argument is used in combination with F
            to generate the signal y.
            For BPSK and QPSK modulations the statistics
            of the ensemble are unimpacted by the choice (all choices are equivalent
            subject to spin-reversal transform). If the argument is None, symbols are
            chosen as 1 or 1 + 1j for all users, respectively for BPSK and QPSK.
            For QAM modulations, amplitude randomness impacts the likelihood in a 
            non-trivial way. If the argument is None in these cases, symbols are
            chosen i.i.d. from the appropriate constellation. Note that, for correct
            analysis of some solvers in BPSK and QPSK cases it is necessary to apply 
            a spin-reversal transform.

        F_distribution:
           When F is None, this argument describes the zero-mean variance 1 
           distribution used to sample each element in F. Permitted values are 
           'Normal' and 'Binary'. For large bandwidth and number of users the
           statistical properties of the likelihood are weakly dependent on this
           choice. When binary is chosen, couplers are integer valued.

        use_offset:
           When True, a constant is added to the Ising model energy so that
           the energy evaluated for the transmitted symbols is zero. At sufficiently
           high bandwidth/user ratio, and signal to noise ratio, this will
           be the ground state energy with high probability.

    Returns:
        The binary quadratic model defining the log-likelihood function

    Example:

        Generate an instance of a CDMA problem in the high-load regime, near a first order
        phase transition _[#T02,#R20]:

        >>> num_variables = 64
        >>> var_per_bandwith = 1.4
        >>> SNR = 5
        >>> bqm = dimod.generators.random_nae3sat(modulation='BPSK', num_var = 64, \
                      bandwidth = round(num_var*var_per_bandwidth), \
                      SNR=SNR, \
                      F_distribution = 'Binary')

         
    .. [#T02] T. Tanaka IEEE TRANSACTIONS ON INFORMATION THEORY, VOL. 48, NO. 11, NOVEMBER 2002
    .. [#R20] J. Raymond, N. Ndiaye, G. Rayaprolu and A. D. King, "Improving performance of logical qubits by parameter tuning and topology compensation," 2020 IEEE International Conference on Quantum Computing and Engineering (QCE), Denver, CO, USA, 2020, pp. 295-305, doi: 10.1109/QCE49297.2020.00044.
    .. [#Prince] Various (https://paws.princeton.edu/) 
    """
    
    if F is not None and y is not None:
        pass
    else:
        if num_var is None:
            if F is not None:
                num_var = F.shape[1]
            else:
                raise ValueError('num_var is not specified and cannot'
                                 'be inferred from F (=None)')
        if bandwidth is None:
            if F is not None:
                bandwidth = F.shape[0]
            elif y is not None:
                bandwidth = y.shape[0]
            else:
                raise ValueError('bandwidth is not specified and cannot'
                                 'be inferred from F or y (both None)')

        random_state = np.random.RandomState(seed)
        assert num_var > 0, "Expect channel users"
        assert bandwidth > 0, "Expect channel users"
        if F is None:
            if F_distribution == 'Binary':
                F = (1-2*random_state.randint(2,size=(bandwidth,num_var)));
            elif F_distribution == 'Normal':
                F = random_state.normal(0,1,size=(bandwidth,num_var));
        if y is None:
            assert SNR > 0, "Expect positive signal to noise ratio"
            if modulation == '16QAM':
                amps = np.arange(-3,5,2)
            elif modulation == '64QAM':
                amps = np.arange(-7,9,2)
            else:
                amps = 1
            sigma = 1/np.sqrt(2*SNR/np.mean(amps*amps));
            
            if transmitted_symbols is None:
                if modulation == 'BPSK':
                    transmitted_symbols = np.ones(shape=(num_var,1))
                elif modulation == 'QPSK': 
                    transmitted_symbols = np.ones(shape=(num_var,1)) \
                                          + 1j*np.ones(shape=(num_var,1))
                else:
                    transmitted_symbols = np.random.choice(amps,size=(num_var,1))
            if modulation == 'BPSK':
                channel_noise = sigma*random_state.normal(0,1,size=(bandwidth,1));
            else:
                channel_noise = sigma*(random_state.normal(0,1,size=(bandwidth,1)) \
                                + 1j*random_state.normal(0,1,size=(bandwidth,1)));
            y = channel_noise + np.matmul(F,transmitted_symbols)
            #print('y',y,'F',F,'transmitted_symbols',transmitted_symbols)
    offset, h, J = _quadratic_form(y,F)
    h, J = _real_quadratic_form(h,J)
    h, J = _amplitude_modulated_quadratic_form(h,J,modulation)
    if use_offset:
        return dimod.BQM(h[:,0],J,'SPIN',offset=offset)
    else:
        np.fill_diagonal(J,0)
        return dimod.BQM(h[:,0],J,'SPIN')
    
def cdma(num_var: int = 64,  var_per_unit_bandwidth: float = 1.4, SNR: float = 5, *,
         discreteSS: bool = True, random_state: Union[None,np.random.RandomState, int] = None,
         noise_discretization: float = None, constellation: str = 'BPSK',
         planted_state: Iterable = None, offset: bool = True) -> tuple:
    """Generate a cdma ensemble problem over a Gaussian channel.
 
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
    .. [#] Various (https://paws.princeton.edu/) 
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
        spreading_sequence = random_state.normal(0,1,size=(bandwidth,num_var));
    
    spreading_sequence_scale = np.sqrt(num_var)
    
    white_gaussian_noise = spreading_sequence_scale*random_state.normal(0,sigma,(bandwidth,1))

    if constellation != 'BPSK':
        #Need imaginary part
        if discreteSS:
            spreading_sequenceI = (1-2*random_state.randint(2,size=(bandwidth,num_var)));
        else:
            spreading_sequenceI = random_state.normal(0,1,size=(bandwidth,num_var));
        white_gaussian_noiseI = spreading_sequence_scale*random_state.normal(0,sigma,size=(bandwidth,1))
    
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
            #All 1 is sufficiently general for SRT invariant solvers:
            transmitted_symbols = np.ones(shape=(num_var,1))
            transmitted_symbolsI = transmitted_symbols
        else:
            #By default, need to use random values:
            transmitted_symbols = random_state.choice([-3,-1,1,3],size=(num_var,1))
            transmitted_symbolsI = random_state.choice([-3,-1,1,3],size=(num_var,1))
    else:
        if constellation == 'BPSK' and len(planted_state) != num_var:
            raise ValueError('planted state is wrong length, should be iterable of num_var real values')
        else:
            if len(planted_state) != 2*num_var:
                raise ValueError('planted state is wrong length, should be iterable of 2*num_var real values')
            transmitted_symbolsI = np.array([[planted_state[i]] for i in range(num_var, 2*num_var)], dtype=float)
        transmitted_symbols = np.array([[planted_state[i]] for i in range(num_var)], dtype=float)
        
    # BPSK (real-real) part
    signal = np.matmul(spreading_sequence,transmitted_symbols) + white_gaussian_noise # JR pR + nR
    E0 = sum(signal*signal)
    J = np.matmul(spreading_sequence.T,spreading_sequence) # FR FR
    h = - 2*np.matmul(spreading_sequence.T,signal) # - 2 FR yR
    if constellation != 'BPSK':
        # See https://confluence.dwavesys.com/display/~jraymond/QPSK+and+16QAM+MIMO
        # [Real Mixed; Mixed Imag]
        signal -= np.matmul(spreading_sequenceI,transmitted_symbolsI) # -JI pI + JR pR + nR
        signalI = np.matmul(spreading_sequence,transmitted_symbolsI) \
                  + np.matmul(spreading_sequenceI,transmitted_symbols) \
                  + white_gaussian_noiseI; # JI pR + JR pI + nR
        E0 += sum(signalI*signalI)

        h -= 2*np.matmul(spreading_sequenceI.T,signalI) #: - 2 FR yR - 2 FI yI
        h = np.concatenate((h, 2*np.matmul(spreading_sequenceI.T,signal) - 2*np.matmul(spreading_sequence.T,signalI)), axis=0) #: - 2 FR yR - 2 FI yI ; 2 FI yR - 2 FR yI

        J += np.matmul(spreading_sequenceI.T,spreading_sequenceI) # FR FR + FI FI
        J_block_topright = - np.matmul(spreading_sequence.T,spreading_sequenceI) \
                           + np.matmul(spreading_sequenceI.T,spreading_sequence) # - FR FI + FI FR
        J = np.concatenate((np.concatenate((J, J_block_topright), axis=1),
                            np.concatenate((J_block_topright.T, J), axis=1)),
                            axis=0) #
        
        if constellation == '16QAM':
            # Outer product under linear encoding:
            h = np.kron(h, np.array([[1],[2]],dtype=float))
            J = np.kron(J, np.array([[1,2],[2,4]],dtype=float))

    
    if SNR < float('Inf'):
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
    else:
        natural_scale = float('Inf')
    
    couplingDict = {}
    hDict = {}
    for u in range(num_var):
        hDict[u] = h[u]
        for v in range(u+1,num_var):
            couplingDict[(u,v)] = J[u][v] + J[v][u]
            assert J[u][v] == J[v][u]
        E0 += J[u][u]
    if offset ==True:
        offset = E0
    else:
        offset = 0
    bqm = dimod.BinaryQuadraticModel(hDict, couplingDict, offset=E0, vartype=dimod.Vartype.SPIN)
    return bqm, random_state, natural_scale, E0
