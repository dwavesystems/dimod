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
    
