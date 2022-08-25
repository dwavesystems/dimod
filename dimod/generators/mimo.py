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
from itertools import product
from typing import Callable, Sequence, Union, Iterable

def _quadratic_form(y, F):
    '''Convert O(v) = ||y - F v||^2 to a sparse quadratic form, where
    y, F are assumed to be complex or real valued.

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
        raise ValueError('y should have shape [n, 1] for some n')
    if len(F.shape) != 2 or F.shape[0] != y.shape[0]:
        raise ValueError('F should have shape [n, m] for some m, n'
                         'and n should equal y.shape[1]')

    offset = np.matmul(y.imag.T, y.imag) + np.matmul(y.real.T, y.real)
    h = - 2*np.matmul(F.T.conj(), y) ## Be careful with interpretaion!
    J = np.matmul(F.T.conj(), F) 

    return offset, h, J

def _real_quadratic_form(h, J, modulation=None):
    '''Unwraps objective function on complex variables onto objective
    function of concatenated real variables: the real and imaginary
    parts.
    '''
    if modulation != 'BPSK' and (h.dtype == np.complex128 or J.dtype == np.complex128):
        hR = np.concatenate((h.real, h.imag), axis=0)
        JR = np.concatenate((np.concatenate((J.real, J.imag), axis=0), 
                            np.concatenate((J.imag.T, J.real), axis=0)), 
                           axis=1)
        return hR, JR
    else:
        return h.real, J.real

def _amplitude_modulated_quadratic_form(h, J, modulation):
    if modulation == 'BPSK' or modulation == 'QPSK':
        #Easy case, just extract diagonal
        return h, J
    else:
        #Quadrature + amplitude modulation
        if modulation == '16QAM':
            num_amps = 2
        elif modulation == '64QAM':
            num_amps = 3
        else:
            raise ValueError('unknown modulation')
        amps = 2**np.arange(num_amps)
        hA = np.kron(amps[:, np.newaxis], h)
        JA = np.kron(np.kron(amps[:, np.newaxis], amps[np.newaxis, :]), J)
        return hA, JA 

    
    
def symbols_to_spins(symbols: np.array, modulation: str) -> np.array:
    "Converts binary/quadrature amplitude modulated symbols to spins, assuming linear encoding"
    num_transmitters = len(symbols)
    if modulation == 'BPSK':
        return symbols.copy()
    else:
        if modulation == 'QPSK':
            # spins_per_real_symbol = 1
            return np.concatenate((symbols.real, symbols.imag))
        elif modulation == '16QAM':
            spins_per_real_symbol = 2
        elif modulation == '64QAM':
            spins_per_real_symbol = 3
        else:
            raise ValueError('Unsupported modulation')
        # A map from integer parts to real is clearest (and sufficiently performant), 
        # generalizes to gray code more easily as well:
        
        symb_to_spins = { np.sum([x*2**xI for xI, x in enumerate(spins)]) : spins
                          for spins in product(*[(-1, 1) for x in range(spins_per_real_symbol)])}
        spins = np.concatenate([np.concatenate(([symb_to_spins[symb][prec] for symb in symbols.real], 
                                                [symb_to_spins[symb][prec] for symb in symbols.imag]))
                                for prec in range(spins_per_real_symbol)])
        
    return spins

def spins_to_symbols(spins: np.array, modulation: str = None, num_transmitters: int = None) -> np.array:
    "Converts spins to modulated symbols assuming a linear encoding"
    num_spins = len(spins)
    if num_transmitters is None:
        if modulation == 'BPSK':
            num_transmitters = num_spins
        elif modulation == 'QPSK':
            num_transmitters = num_spins//2
        elif modulation == '16QAM':
            num_transmitters = num_spins//4
        elif modulation == '64QAM':
            num_transmitters = num_spins//6
        else:
            raise ValueError('Unsupported modulation')
        
    if num_transmitters == num_spins:
        symbols = spins 
    else:
        num_amps, rem = divmod(len(spins), (2*num_transmitters))
        if num_amps > 64:
            raise ValueError('Complex encoding is limited to 64 bits in'
                             'real and imaginary parts; num_transmitters is'
                             'too small')
        if rem != 0:
            raise ValueError('num_spins must be divisible by num_transmitters '
                             'for modulation schemes')
        
        spinsR = np.reshape(spins, (num_amps, 2*num_transmitters))
        amps = 2**np.arange(0, num_amps)[:, np.newaxis]
        
        symbols = np.sum(amps*spinsR[:, :num_transmitters], axis=0) \
                + 1j * np.sum(amps*spinsR[:, num_transmitters:], axis=0)
    return symbols

def _create_channel(random_state, num_receivers, num_transmitters, F_distribution):
    """Create a channel model"""
    channel_power = 1
    if F_distribution is None:
        F_distribution = ('Normal', 'Complex')
    elif type(F_distribution) is not tuple or len(F_distribution) !=2:
        raise ValueError('F_distribution should be a tuple of strings or None')
    if F_distribution[0] == 'Normal':
        if F_distribution[1] == 'Real':
            F = random_state.normal(0, 1, size=(num_receivers, num_transmitters))
        else:
            F = random_state.normal(0, 1, size=(num_receivers, num_transmitters)) + 1j*random_state.normal(0, 1, size=(num_receivers, num_transmitters))
    elif F_distribution[0] == 'Binary':
        if modulation == 'BPSK':
            F = (1-2*random_state.randint(2, size=(num_receivers, num_transmitters)))
        else:
            channel_power = 2 #For integer precision purposes:
            F = (1-2*random_state.randint(2, size=(num_receivers, num_transmitters))) + 1j*(1-2*random_state.randint(2, size=(num_receivers, num_transmitters)))
    return F, channel_power

def _create_signal(random_state, num_receivers, num_transmitters, SNRb, F, channel_power, modulation, transmitted_symbols):
    assert SNRb > 0, "Expect positive signal to noise ratio"
    
    if modulation == 'BPSK':
        bits_per_transmitter = 1
        amps = 1
    else:
        bits_per_transmitter = 2
        if modulation == '16QAM':
            amps = np.arange(-3, 5, 2)
            bits_per_transmitter *= 2
        elif modulation == '64QAM':
            amps = np.arange(-7, 9, 2)
            bits_per_transmitter *= 3
        else:
            amps = 1
            
    # Energy_per_bit_per_receiver (assuming N0 = 1, for SNRb conversion):
    expectation_Fv = channel_power*np.mean(amps*amps)/bits_per_transmitter
    # Eb/N0 = SNRb/2 (N0 = 2 sigma^2, the one-sided PSD ~ kB T at antenna)
    sigma = np.sqrt(expectation_Fv/(2*SNRb));
            
    if transmitted_symbols is None:
        if modulation == 'BPSK':
            transmitted_symbols = np.ones(shape=(num_transmitters, 1))
        elif modulation == 'QPSK': 
            transmitted_symbols = np.ones(shape=(num_transmitters, 1)) \
                                    + 1j*np.ones(shape=(num_transmitters, 1))
        else:
            transmitted_symbols = np.random.choice(amps, size=(num_transmitters, 1))
    if modulation == 'BPSK' and F.dtype==np.float64:
        #Channel noise is always complex, but only real part is relevant to real channel + real symbols 
        channel_noise = sigma*random_state.normal(0, 1, size=(num_receivers, 1));
    else:
        channel_noise = sigma*(random_state.normal(0, 1, size=(num_receivers, 1)) \
                               + 1j*random_state.normal(0, 1, size=(num_receivers, 1)));
    y = channel_noise + np.matmul(F, transmitted_symbols)
    return y

def _yF_to_hJ(y, F, modulation):
    offset, h, J = _quadratic_form(y, F) # Quadratic form re-expression
    h, J = _real_quadratic_form(h, J, modulation) # Complex symbols to real symbols (if necessary)
    h, J = _amplitude_modulated_quadratic_form(h, J, modulation) # Real symbol to linear spin encoding
    return h, J, offset

def spin_encoded_mimo(modulation: str, y: np.array = None, F: np.array = None,  
                      *, 
                      num_transmitters: int = None,  num_receivers: int = None, SNRb: float = float('Inf'), 
                      seed: Union[None, int, np.random.RandomState] = None, 
                      transmitted_symbols: Iterable = None, 
                      F_distribution: Union[None, str] = None, 
                      use_offset: bool = False) -> dimod.BinaryQuadraticModel:
    """ Generate a multi-input multiple-output (MIMO) channel-decoding problem.
        
    Users each transmit complex valued symbols over a random channel :math:`F` of 
    some num_receivers, subject to additive white Gaussian noise. Given the received
    signal y the log likelihood of a given symbol set :math:`v` is given by 
    :math:`MLE = argmin || y - F v ||_2`. When v is encoded as a linear
    sum of spins the optimization problem is defined by a Binary Quadratic Model. 
    Depending on arguments used, this may be a model for Code Division Multiple
    Access _[#T02, #R20], 5G communication network problems _[#Prince], or others.
    
    Args:
        y: A complex or real valued signal in the form of a numpy array. If not
            provided, generated from other arguments.

        F: A complex or real valued channel in the form of a numpy array. If not
            provided, generated from other arguments.

        modulation: Specifies the constellation (symbol set) in use by 
            each user. Symbols are assumed to be transmitted with equal probability.
            Options are:
               * 'BPSK'
                   Binary Phase Shift Keying. Transmitted symbols are +1, -1;
                   no encoding is required.
                   A real valued channel is assumed.

               * 'QPSK'
                   Quadrature Phase Shift Keying. 
                   Transmitted symbols are +1, -1, +1j, -1j;
                   spins are encoded as a real vector concatenated with an imaginary vector.
                   
               * '16QAM'
                   Each user is assumed to select independently from 16 symbols.
                   The transmitted symbol is a complex value that can be encoded by two spins
                   in the imaginary part, and two spins in the real part. v = 2 s_1 + s_2.
                   Highest precision real and imaginary spin vectors, are concatenated to 
                   lower precision spin vectors.
                   
               * '64QAM'
                   A QPSK symbol set is generated, symbols are further amplitude modulated 
                   by an independently and uniformly distributed random amount from [1, 3].

        num_transmitters: Number of users. Since each user transmits 1 symbol per frame, also the
             number of transmitted symbols, must be consistent with F argument.

        num_receivers: Num_Receivers of channel, :code:`len(y)`. Must be consistent with y argument.

        SNRb: Signal to noise ratio per bit on linear scale. When y is not provided, this is used
            to generate the noisy signal. In the case float('Inf') no noise is 
            added. SNRb = Eb/N0, where Eb is the energy per bit, and N0 is the one-sided
            power-spectral density. A one-sided . N0 is typically kB T at the receiver. 
            To convert units of dB to SNRb use SNRb=10**(SNRb[decibells]/10).
        
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
           distribution used to sample each element in F. Permitted values are in
           tuple form: (str, str). The first string is either 
           'Normal' or 'Binary'. The second string is either 'Real' or 'Complex'.
           For large num_receivers and number of users the statistical properties of 
           the likelihood are weakly dependent on the first argument. Choosing 
           'Binary' allows for integer valued Hamiltonians, 'Normal' is a more 
           standard model. The channel can be Real or Complex. In many cases this 
           also represents a superficial distinction up to rescaling. For real 
           valued symbols (BPSK) the default is ('Normal', 'Real'), otherwise it
           is ('Normal', 'Complex')

        use_offset:
           When True, a constant is added to the Ising model energy so that
           the energy evaluated for the transmitted symbols is zero. At sufficiently
           high num_receivers/user ratio, and signal to noise ratio, this will
           be the ground state energy with high probability.

    Returns:
        The binary quadratic model defining the log-likelihood function

    Example:

        Generate an instance of a CDMA problem in the high-load regime, near a first order
        phase transition _[#T02, #R20]:

        >>> num_transmitters = 64
        >>> var_per_bandwith = 1.4
        >>> SNR = 5
        >>> bqm = dimod.generators.random_nae3sat(modulation='BPSK', num_transmitters = 64, \
                      num_receivers = round(num_transmitters*var_per_num_receivers), \
                      SNR=SNR, \
                      F_distribution = 'Binary')

         
    .. [#T02] T. Tanaka IEEE TRANSACTIONS ON INFORMATION THEORY, VOL. 48, NO. 11, NOVEMBER 2002
    .. [#R20] J. Raymond, N. Ndiaye, G. Rayaprolu and A. D. King, "Improving performance of logical qubits by parameter tuning and topology compensation, " 2020 IEEE International Conference on Quantum Computing and Engineering (QCE), Denver, CO, USA, 2020, pp. 295-305, doi: 10.1109/QCE49297.2020.00044.
    .. [#Prince] Various (https://paws.princeton.edu/) 
    """
    
    if F is not None and y is not None:
        pass
    else:
        if num_transmitters is None:
            if F is not None:
                num_transmitters = F.shape[1]
            elif transmitted_symbols is not None:
                num_transmitters = len(transmitted_symbols)
            else:
                raise ValueError('num_transmitters is not specified and cannot'
                                 'be inferred from F or transmitted_symbols (both None)')
        if num_receivers is None:
            if F is not None:
                num_receivers = F.shape[0]
            elif y is not None:
                num_receivers = y.shape[0]
            else:
                raise ValueError('num_receivers is not specified and cannot'
                                 'be inferred from F or y (both None)')

        random_state = np.random.RandomState(seed)
        assert num_transmitters > 0, "Expect positive number of transmitters"
        assert num_receivers > 0, "Expect positive number of receivers"
        
        F, channel_power = _create_channel(random_state, num_receivers,
                                           num_transmitters, F_distribution)
       
        if y is None:
            y = _create_signal(random_state, num_receivers, num_transmitters,
                               SNRb, F, channel_power, modulation, transmitted_symbols)

    h, J, offset = _yF_to_hJ(y, F, modulation)
  
    if use_offset:
        return dimod.BQM(h[:,0], J, 'SPIN', offset=offset)
    else:
        np.fill_diagonal(J, 0)
        return dimod.BQM(h[:,0], J, 'SPIN')
    
    
