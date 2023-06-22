# -*- coding: utf-8 -*-
# Copyright 2023 D-Wave Systems Inc.
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

from itertools import product
import networkx as nx
import numpy as np
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

import dimod 

def _quadratic_form(y, F):
    """Convert :math:`O(v) = ||y - F v||^2` to sparse quadratic form.
    
    Constructs coefficients for the form 
    :math:`O(v) = v^{\dagger} J v - 2 \Re(h^{\dagger} v) + \\text{offset}`. 

    Args:
        y: Received symbols as a NumPy column vector of complex or real values.

        F: Wireless channel as an :math:`i \\times j` NumPy matrix of complex 
            values, where :math:`i` rows correspond to :math:`y_i` receivers 
            and :math:`j` columns correspond to :math:`v_i` transmitted symbols.
    
    Returns:
        Three tuple of offset, as a real scalar, linear biases :math:`h`, as a dense
        real vector, and quadratic interactions, :math:`J`, as a dense real symmetric 
        matrix.
    """
    if len(y.shape) != 2 or y.shape[1] != 1:
        raise ValueError(f"y should have shape (n, 1) for some n; given: {y.shape}")
    
    if len(F.shape) != 2 or F.shape[0] != y.shape[0]:
        raise ValueError("F should have shape (n, m) for some m, n "
                         "and n should equal y.shape[1];" 
                         f" given: {F.shape}, n={y.shape[1]}")

    offset = np.matmul(y.imag.T, y.imag) + np.matmul(y.real.T, y.real)
    h = - 2*np.matmul(F.T.conj(), y)    # Be careful with interpretation!
    J = np.matmul(F.T.conj(), F) 

    return offset, h, J

def _real_quadratic_form(h, J, modulation=None):
    """Separate real and imaginary parts of quadratic form.
    
    Unwraps objective function on complex variables as an objective function of 
    concatenated real variables, first the real and then the imaginary part.

    Args:
        h: Linear biases as a dense real NumPy vector. 
        
        J: Quadratic interactions as a dense real symmetric matrix.

        modulation: Modulation. Supported values are 'BPSK', 'QPSK', '16QAM', 
            '64QAM', and '256QAM'.

    Returns:
        Two-tuple of linear biases, :math:`h`, as a NumPy real vector with any 
        imaginary part following the real part, and quadratic interactions, 
        :math:`J`, as a real matrix with any imaginary part moved to above and 
        below the diagonal.
    """
    # Here, for BPSK F-induced complex parts of h and J are discarded:  
    # Given y = F x + nu, for independent and identically distributed channel F 
    # and complex noise nu, the system of equations defined by the real part is 
    # sufficient to define the canonical decoding problem. 
    # In essence, rotate y to the eigenbasis of F, throw away the orthogonal noise 
    # (the complex problem as the real part with suitable adjustment factor 2 to 
    # signal to noise ratio: F^{-1}*y = I*x + F^{-1}*nu)
    # JR: revisit and prove

    if modulation != 'BPSK' and (any(np.iscomplex(h)) or any(np.iscomplex(J))):
        hR = np.concatenate((h.real, h.imag), axis=0)
        JR = np.concatenate((np.concatenate((J.real, J.imag), axis=0), 
                             np.concatenate((J.imag.T, J.real), axis=0)), 
                             axis=1)
        return hR, JR
    else:
        return h.real, J.real

def _amplitude_modulated_quadratic_form(h, J, modulation):
    """Amplitude-modulate the quadratic form.
    
    Updates bias amplitudes for quadrature amplitude modulation.

    Args:
        h: Linear biases as a NumPy vector. 
        
        J: Quadratic interactions as a matrix.

        modulation: Modulation. Supported values are non-quadrature modulation 
            BPSK and quadrature modulations 'QPSK', '16QAM', '64QAM', and '256QAM'.

    Returns:
        Two-tuple of amplitude-modulated linear biases, :math:`h`, as a NumPy 
        vector  and amplitude-modulated quadratic interactions, :math:`J`, as 
        a matrix.
    """
    if modulation == 'BPSK' or modulation == 'QPSK':
        #Easy case, just extract diagonal
        return h, J
    else:
        # Quadrature + amplitude modulation
        if modulation == '16QAM':
            num_amps = 2
        elif modulation == '64QAM':
            num_amps = 3
        else:
            raise ValueError('unknown modulation')
        
        amps = 2 ** np.arange(num_amps)
        hA = np.kron(amps[:, np.newaxis], h)
        JA = np.kron(np.kron(amps[:, np.newaxis], amps[np.newaxis, :]), J)
        return hA, JA 
      
def _symbols_to_spins(symbols: np.array, modulation: str) -> np.array:
    """Convert quadrature amplitude modulated (QAM) symbols to spins. 

    Args:
        symbols: Transmitted symbols as a NumPy column vector. 
 
        modulation: Modulation. Supported values are non-quadrature modulation 
            binary phase-shift keying (BPSK, or 2-QAM) and quadrature modulations 
            'QPSK', '16QAM', '64QAM', and '256QAM'.

    Returns:
        Spins as a NumPy array.    
    """
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
        # generalizes to Gray coding more easily as well:
        
        symb_to_spins = { np.sum([x*2**xI for xI, x in enumerate(spins)]) : spins
                          for spins in product(*spins_per_real_symbol*[(-1, 1)])}
        spins = np.concatenate([np.concatenate(([symb_to_spins[symb][prec] for symb in symbols.real.flatten()], 
                                                [symb_to_spins[symb][prec] for symb in symbols.imag.flatten()]))
                                for prec in range(spins_per_real_symbol)])
        if len(symbols.shape) > 2:
            raise ValueError(f"`symbols` should be 1 or 2 dimensional but is shape {symbols.shape}")
        if symbols.ndim == 1:    # If symbols shaped as vector, return as vector
            spins.reshape((len(spins), ))

    return spins

def _yF_to_hJ(y, F, modulation):
    offset, h, J = _quadratic_form(y, F) # Quadratic form re-expression
    # Complex to real symbols (if necessary): 
    h, J = _real_quadratic_form(h, J, modulation) 
    # Real symbol to linear spin encoding:
    h, J = _amplitude_modulated_quadratic_form(h, J, modulation) 
    return h, J, offset

def linear_filter(F, method='zero_forcing', SNRoverNt=float('Inf'), PoverNt=1):
    """ Construct linear filter W for estimation of transmitted signals.
    # https://www.youtube.com/watch?v=U3qjVgX2poM
   
    
    We follow conventions laid out in MacKay et al. 'Achievable sum rate of MIMO 
    MMSE receivers: A general analytic framework'
    N0 Identity[N_r] = E[n n^dagger]
    P/N_t Identify[N_t] = E[v v^dagger], i.e. P = constellation_mean_power*Nt for 
    i.i.d elements (1,2,10,42)Nt for BPSK, QPSK, 16QAM, 64QAM.
    N_r N_t = E_F[Tr[F Fdagger]], i.e. E[||F_{mu,i}||^2]=1 for i.i.d channel.  
    normalization is assumed to be pushed into symbols.
    SNRoverNt = PoverNt/N0 : Intensive quantity. 
    SNRb = SNR/(Nt*bits_per_symbol)

    Typical use case: set SNRoverNt = SNRb
    """
    
    if method == 'zero_forcing':
        # Moore-Penrose pseudo inverse
        W = np.linalg.pinv(F)
    else:
        Nr, Nt = F.shape
         # Matched Filter
        if method == 'matched_filter':
            W = F.conj().T / np.sqrt(PoverNt)
            # F = root(Nt/P) Fcompconj
        elif method == 'MMSE':
            W = np.matmul(
                F.conj().T, 
                np.linalg.pinv(np.matmul(F, F.conj().T) + np.identity(Nr)/SNRoverNt)
                         ) / np.sqrt(PoverNt)
        else:
            raise ValueError('Unsupported linear method')
    return W
    
def filter_marginal_estimator(x: np.array, modulation: str):
    if modulation is not None:
        if modulation == 'BPSK' or modulation == 'QPSK':
            max_abs = 1
        elif modulation == '16QAM':
            max_abs = 3
        elif modulation == '64QAM':
            max_abs = 7
        elif modulation == '128QAM':
            max_abs = 15
        else:
            raise ValueError('Unknown modulation')
        #Real part (nearest):
        x_R = 2*np.round((x.real - 1)/2) + 1
        x_R = np.where(x_R < -max_abs, -max_abs, x_R)
        x_R = np.where(x_R > max_abs, max_abs, x_R)
        if modulation != 'BPSK':
            x_I = 2*np.round((x.imag - 1)/2) + 1
            x_I = np.where(x_I <- max_abs, -max_abs, x_I)
            x_I = np.where(x_I > max_abs, max_abs, x_I)
            return x_R + 1j*x_I
        else:
            return x_R
        
def spins_to_symbols(spins: np.array, modulation: str = None, 
                     num_transmitters: int = None) -> np.array:
    """Converts spins to modulated symbols assuming a linear encoding"""
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
                + 1j*np.sum(amps*spinsR[:, num_transmitters:], axis=0)
    return symbols

def lattice_to_attenuation_matrix(lattice,transmitters_per_node=1,receivers_per_node=1,neighbor_root_attenuation=1):
    """The attenuation matrix is an ndarray and specifies the expected root-power of transmission between integer indexed transmitters and receivers.
    The shape of the attenuation matrix is num_receivers by num_transmitters.
    In this code, there is uniform transmission of power for on-site trasmitter/receiver pairs, and unifrom transmission
    from transmitters to receivers up to graph distance 1.
    Note that this code requires work - we should exploit sparsity, and track the label map.
    This could be generalized to account for asymmetric transmission patterns, or real-valued spatial structure."""
    num_var = lattice.number_of_nodes()

    if any('num_transmitters' in lattice.nodes[n] for n in lattice.nodes) or any('num_receivers' in lattice.nodes[n] for n in lattice.nodes):
        node_to_transmitters = {} #Integer labels of transmitters at node
        node_to_receivers = {} #Integer labels of receivers at node
        t_ind = 0
        r_ind = 0
        for n in lattice.nodes:
            num = transmitters_per_node
            if 'num_transmitters' in lattice.nodes[n]:
                num = lattice.nodes[n]['num_transmitters']
            node_to_transmitters[n] = list(range(t_ind,t_ind+num))
            t_ind = t_ind + num
            
            num = receivers_per_node
            if 'num_receivers' in lattice.nodes[n]:
                num = lattice.nodes[n]['num_receivers']
            node_to_receivers[n] = list(range(r_ind,r_ind+num))
            r_ind = r_ind + num
        A = np.zeros(shape=(r_ind, t_ind))
        for n0 in lattice.nodes:
            root_receivers = node_to_receivers[n0]
            for r in root_receivers:
                for t in node_to_transmitters[n0]:
                    A[r,t] = 1
                for neigh in lattice.neighbors(n0):
                    for t in node_to_transmitters[neigh]:
                        A[r,t]=neighbor_root_attenuation
    else:
        A = np.identity(num_var)
        # Uniform case:
        node_to_int = {n:idx for idx,n in enumerate(lattice.nodes())}
        for n0 in lattice.nodes:
            root = node_to_int[n0]
            for neigh in lattice.neighbors(n0):
                A[node_to_int[neigh],root]=neighbor_root_attenuation
        A = np.tile(A,(receivers_per_node,transmitters_per_node))
        node_to_receivers = {n: [node_to_int[n]+i*len(node_to_int) for i in range(receivers_per_node)] for n in node_to_int}
        node_to_transmitters = {n: [node_to_int[n]+i*len(node_to_int) for i in range(transmitters_per_node)] for n in node_to_int}
    return A, node_to_transmitters, node_to_receivers

def create_channel(num_receivers: int = 1, num_transmitters: int = 1, 
                   F_distribution: Optional[Tuple[str, str]] = None, 
                   random_state: Optional[Union[int, np.random.mtrand.RandomState]] = None, 
                   attenuation_matrix: Optional[np.ndarray] = None) -> Tuple[
                       np.ndarray, float, np.random.mtrand.RandomState]:
    """Create a channel model. 

    Channel power is the expected root mean square signal per receiver; i.e., 
    :math:`mean(F^2)*num_transmitters` for homogeneous codes.

    Args:
        num_receivers: Number of receivers.

        num_transmitters: Number of transmitters.

        F_distribution: Distribution for the channel. Supported values are:

        *   First value: ``normal`` and ``binary``.
        *   Second value: ``real`` and ``complex``.
        
        random_state: Seed for a random state or a random state. 

        attenuation_matrix: Root of the power associated with a variable to 
        chip communication ... Jack: what does this represent in the field?
        Joel: This is the root-power part of the matrix F. It basically sparsifies
        F so as to match the lattice transmission structure. The function now
        has some additional branches that make things more explicit.

    Returns:
        Three-tuple of channel, channel power, and the random state used, where 
        the channel is an :math:`i \times j` matrix with :math:`i` rows 
        corresponding to the receivers and :math:`j` columns to the transmitters, 
        and channel power is a number. 

    """

    if num_receivers < 1 or num_transmitters < 1:
        raise ValueError('At least one receiver and one transmitter are required.')
    #random_state = np.random.RandomState(10) ##DEBUG
    channel_power = num_transmitters
    if not random_state:
        random_state = np.random.RandomState(10)
    elif type(random_state) is not np.random.mtrand.RandomState:
        random_state = np.random.RandomState(random_state) 

    if F_distribution is None:
        F_distribution = ('normal', 'complex')    
    elif type(F_distribution) is not tuple or len(F_distribution) != 2:
        raise ValueError('F_distribution should be a tuple of strings or None')
    
    if F_distribution[0] == 'normal':
        if F_distribution[1] == 'real':
            F = random_state.normal(0, 1, size=(num_receivers, num_transmitters))
        else:
            channel_power = 2*num_transmitters
            F = random_state.normal(0, 1, size=(num_receivers, num_transmitters)) + \
                1j*random_state.normal(0, 1, size=(num_receivers, num_transmitters))
    elif F_distribution[0] == 'binary':
        if F_distribution[1] == 'real':
            F = (1 - 2*random_state.randint(2, size=(num_receivers, num_transmitters)))
        else:
            channel_power = 2*num_transmitters #For integer precision purposes:
            F = (1 - 2*random_state.randint(2, size=(num_receivers, num_transmitters))) + \
                1j*(1 - 2*random_state.randint(2, size=(num_receivers, num_transmitters)))
            
    if attenuation_matrix is not None:
        if np.iscomplex(attenuation_matrix).any():
            raise ValueError('attenuation_matrix must not have complex values')
        F = F*attenuation_matrix #Dense format for now, this is slow.
        channel_power *= np.mean(np.sum(attenuation_matrix*attenuation_matrix, axis=0))

    return F, channel_power, random_state
    
constellation = {   # bits per transmitter (bpt) and amplitudes (amps)
    "BPSK": [1, np.ones(1)],       
    "QPSK": [2, np.ones(1)],
    "16QAM": [4, 1+2*np.arange(2)],
    "64QAM": [6, 1+2*np.arange(4)],
    "256QAM": [8, 1+2*np.arange(8)]} 

def _constellation_properties(modulation):
    """Return bits per symbol, symbol amplitudes, and mean power for QAM constellation. 
    
    Constellation mean power makes the standard assumption that symbols are 
    sampled uniformly at random for the signal.
    """

    bpt_amps = constellation.get(modulation)
    if not bpt_amps:
        raise ValueError('Unsupported modulation method')
    
    constellation_mean_power = 1 if modulation == 'BPSK' else 2*np.mean(bpt_amps[1]*bpt_amps[1]) 

    return bpt_amps[0], bpt_amps[1], constellation_mean_power 

def _create_transmitted_symbols(num_transmitters, 
                                amps=[-1, 1], 
                                quadrature=True, 
                                random_state=None):
    """Generate symbols.

    Symbols are generated uniformly at random as a function of the quadrature 
    and amplitude modulation. 
    
    The power per symbol is not normalized, it is proportional to :math:`N_t*sig2`, 
    where :math:`sig2 = [1, 2, 10, 42]` for BPSK, QPSK, 16QAM and 64QAM respectively. 
    
    The complex and real-valued parts of all constellations are integer.

    Args:
        num_transmitters: Number of transmitters.

        amps: Amplitudes as an interable. 

        quadrature: Quadrature (True) or only phase-shift keying such as BPSK (False).

        random_state: Seed for a random state or a random state.

    Returns:

        Two-tuple of symbols and the random state used, where the symbols is 
        a column vector of length ``num_transmitters``.
    
    """

    if any(np.iscomplex(amps)):
        raise ValueError('Amplitudes cannot have complex values')
    if any(np.modf(amps)[0]):
        raise ValueError('Amplitudes must have integer values')
    
    if type(random_state) is not np.random.mtrand.RandomState:
        random_state = np.random.RandomState(random_state)
    
    if quadrature == False:
        transmitted_symbols = random_state.choice(amps, size=(num_transmitters, 1))
    else: 
        transmitted_symbols = random_state.choice(amps, size=(num_transmitters, 1)) \
            + 1j*random_state.choice(amps, size=(num_transmitters, 1))
        
    return transmitted_symbols, random_state

def _create_signal(F, transmitted_symbols=None, channel_noise=None,
                  SNRb=float('Inf'), modulation='BPSK', channel_power=None,
                  random_state=None):
    """Create signal y = F v + n. 
    
    Generates random transmitted symbols and noise as necessary. 

    F is assumed to consist of independent and identically distributed (i.i.d) 
    elements such that :math:`F\dagger*F = N_r I[N_t]*cp` where :math:`I` is 
    the identity matrix and :math:`cp` the channel power.

    v are assumed to consist of i.i.d unscaled constellations elements (integer 
    valued in real and complex parts). Mean constellation power dictates a 
    rescaling relative to :math:`E[v v\dagger] = I[Nt]`. 
    
    ``channel_noise`` is assumed, or created, to be suitably scaled. N0 Identity[Nt] =  
    SNRb = /   @jack, please finish this statement; also I removed unused F_norm = 1, v_norm = 1

    Args:
        F: Wireless channel as an :math:`i \times j` matrix of complex values, 
            where :math:`i` rows correspond to :math:`y_i` receivers and :math:`j` 
            columns correspond to :math:`v_i` transmitted symbols. 

        transmitted_symbols: Transmitted symbols as a column vector.

        channel_noise: Channel noise as a complex value.

        SNRb: Signal-to-noise ratio.

        modulation: Modulation. Supported values are 'BPSK', 'QPSK', '16QAM', 
            '64QAM', and '256QAM'.

        channel_power: Channel power. By default, proportional to the number 
            of transmitters. 

        random_state: Seed for a random state or a random state.

    Returns:
        Four-tuple of received signals (``y``), transmitted symbols (``v``), 
        channel noise, and random_state, where ``y`` is a column vector of length
        equal to the rows of ``F``.
    """

    num_receivers = F.shape[0]
    num_transmitters = F.shape[1]

    if not random_state:
        random_state = np.random.RandomState(10)
    elif type(random_state) is not np.random.mtrand.RandomState:
        random_state = np.random.RandomState(random_state)
 
    bits_per_transmitter, amps, constellation_mean_power = _constellation_properties(modulation)

    if transmitted_symbols is not None:
        if modulation == 'BPSK' and any(np.iscomplex(transmitted_symbols)):
            raise ValueError(f"BPSK transmitted signals must be real")
        if modulation != 'BPSK' and any(np.isreal(transmitted_symbols)):
            raise ValueError(f"Quadrature transmitted signals must be complex")
    else:
        if type(random_state) is not np.random.mtrand.RandomState:
            random_state = np.random.RandomState(random_state)
        
        quadrature = False if modulation == 'BPSK' else True
        transmitted_symbols, random_state = _create_transmitted_symbols(
                num_transmitters, amps=amps, quadrature=quadrature, random_state=random_state)
           
    if SNRb <= 0:
       raise ValueError(f"signal-to-noise ratio must be positive. SNRb={SNRb}")
    
    if SNRb == float('Inf'):
       y = np.matmul(F, transmitted_symbols)
    elif channel_noise is not None:
        y = channel_noise + np.matmul(F, transmitted_symbols)
    else:
        # Energy_per_bit:
        if channel_power == None:   
            #Assume proportional to num_transmitters; i.e., every channel component is RMSE 1 and 1 bit
            channel_power = num_transmitters

        Eb = channel_power * constellation_mean_power / bits_per_transmitter #Eb is the same for QPSK and BPSK
        # Eb/N0 = SNRb (N0 = 2 sigma^2, the one-sided PSD ~ kB T at antenna)
        # SNRb and Eb, together imply N0
        N0 = Eb / SNRb
        sigma = np.sqrt(N0/2) # Noise is complex by definition, hence 1/2 power in real and complex parts

        # Channel noise of covariance N0*I_{NR}. Noise is complex by definition, although
        # for real channel and symbols we need only worry about real part:
        channel_noise = sigma*(random_state.normal(0, 1, size=(num_receivers, 1)) \
            + 1j*random_state.normal(0, 1, size=(num_receivers, 1)))
        if modulation == 'BPSK' and np.isreal(F).all():
            channel_noise = channel_noise.real
            
        y = channel_noise + np.matmul(F, transmitted_symbols)

    return y, transmitted_symbols, channel_noise, random_state

# JP: Leave remainder untouched for next PRs to avoid conflicts before this is merged
#     Next PR should bring in commit https://github.com/jackraymond/dimod/commit/ef99d2ae1c364f2066018046a0ece977443b229e

def spin_encoded_mimo(modulation: str, y: Union[np.array, None] = None, F: Union[np.array, None] = None,
                      *,
                      transmitted_symbols: Union[np.array, None] = None, channel_noise: Union[np.array, None] = None, 
                      num_transmitters: int = None,  num_receivers: int = None, SNRb: float = float('Inf'), 
                      seed: Union[None, int, np.random.RandomState] = None, 
                      F_distribution: Union[None, tuple] = None, 
                      use_offset: bool = False,
                      attenuation_matrix = None) -> dimod.BinaryQuadraticModel:
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
           provided, generated from other arguments. Note that for correct interpretation
           of SNRb, the channel power should be normalized to num_transmitters.

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
           'normal' or 'binary'. The second string is either 'real' or 'complex'.
           For large num_receivers and number of users the statistical properties of 
           the likelihood are weakly dependent on the first argument. Choosing 
           'binary' allows for integer valued Hamiltonians, 'normal' is a more 
           standard model. The channel can be real or complex. In many cases this 
           also represents a superficial distinction up to rescaling. For real 
           valued symbols (BPSK) the default is ('normal', 'real'), otherwise it
           is ('normal', 'complex')

        use_offset:
           When True, a constant is added to the Ising model energy so that
           the energy evaluated for the transmitted symbols is zero. At sufficiently
           high num_receivers/user ratio, and signal to noise ratio, this will
           be the ground state energy with high probability.

        attenuation_matrix:
           Root power associated to variable to chip communication; use
           for sparse and structured codes.
    
    Returns:
        The binary quadratic model defining the log-likelihood function

    Example:

        Generate an instance of a CDMA problem in the high-load regime, near a first order
        phase transition _[#T02, #R20]:

        >>> num_transmitters = 64
        >>> transmitters_per_receiver = 1.5
        >>> SNRb = 5
        >>> bqm = dimod.generators.spin_encoded_mimo(modulation='BPSK', num_transmitters = 64, \
                      num_receivers = round(num_transmitters/transmitters_per_receiver), \
                      SNRb=SNRb, \
                      F_distribution = ('binary','real'))

         
    .. [#T02] T. Tanaka IEEE TRANSACTIONS ON INFORMATION THEORY, VOL. 48, NO. 11, NOVEMBER 2002
    .. [#R20] J. Raymond, N. Ndiaye, G. Rayaprolu and A. D. King, "Improving performance of logical qubits by parameter tuning and topology compensation, " 2020 IEEE International Conference on Quantum Computing and Engineering (QCE), Denver, CO, USA, 2020, pp. 295-305, doi: 10.1109/QCE49297.2020.00044.
    .. [#Prince] Various (https://paws.princeton.edu/) 
    """
    
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
        elif channel_noise is not None:
            num_receivers = channel_noise.shape[0]
        else:
            raise ValueError('num_receivers is not specified and cannot'
                             'be inferred from F, y or channel_noise (all None)')

    assert num_transmitters > 0, "Expect positive number of transmitters"
    assert num_receivers > 0, "Expect positive number of receivers"

    if F is None:
        F, channel_power, seed = create_channel(num_receivers=num_receivers, num_transmitters=num_transmitters,
                                                F_distribution=F_distribution, random_state=seed, attenuation_matrix=attenuation_matrix)
        #Channel power is the value relative to an assumed normalization E[Fui* Fui] = 1 
    else:
        channel_power = num_transmitters
       
    if y is None:
        y, _, _, _ = _create_signal(F, transmitted_symbols=transmitted_symbols, channel_noise=channel_noise,
                                   SNRb=SNRb, modulation=modulation, channel_power=channel_power,
                                   random_state=seed)
    
    h, J, offset = _yF_to_hJ(y, F, modulation)
  
    if use_offset:
        #NB - in this form, offset arises from 
        return dimod.BQM(h[:,0], J, 'SPIN', offset=offset)
    else:
        np.fill_diagonal(J, 0)
        return dimod.BQM(h[:,0], J, 'SPIN')

def _make_honeycomb(L: int):
    """ 2L by 2L triangular lattice with open boundaries,
    and cut corners to make hexagon. """
    G = nx.Graph()
    G.add_edges_from([((x, y), (x,y+ 1)) for x in range(2*L+1) for y in range(2*L)])
    G.add_edges_from([((x, y), (x+1, y)) for x in range(2*L) for y in range(2*L + 1)])
    G.add_edges_from([((x, y), (x+1, y+1)) for x in range(2*L) for y in range(2*L)])
    G.remove_nodes_from([(i,j) for j in range(L) for i in range(L+1+j,2*L+1) ])
    G.remove_nodes_from([(i,j) for i in range(L) for j in range(L+1+i,2*L+1)])
    return G


def spin_encoded_comp(lattice: Union[int,nx.Graph],
                      modulation: str, y: Union[np.array, None] = None,
                      F: Union[np.array, None] = None,
                      *,
                      integer_labeling: bool = True,
                      transmitted_symbols: Union[np.array, None] = None, channel_noise: Union[np.array, None] = None, 
                      num_transmitters_per_node: int = 1,
                      num_receivers_per_node: int = 1, SNRb: float = float('Inf'), 
                      seed: Union[None, int, np.random.RandomState] = None, 
                      F_distribution: Union[None, str] = None, 
                      use_offset: bool = False) -> dimod.BinaryQuadraticModel:
    """Defines a simple coooperative multi-point decoding problem CoMP.
    Args:
       lattice: A graph defining the set of nearest neighbor basestations. Each 
           basestation has ``num_receivers`` receivers and ``num_transmitters`` 
           local transmitters. Transmitters from neighboring basestations are also 
           received. The channel F should be set to None, it is not dependent on the
           geometric information for now.
           Node attributes 'num_receivers' and 'num_transmitters' override the 
           input defaults.
           lattice can also be set to an integer value, in which case a honeycomb 
           lattice of the given linear scale (number of basestations O(L^2)) is 
           created using ``_make_honeycomb()``.
       modulation: modulation
       integer_labeling:
           When True, the geometric, quadrature and modulation-scale information
           associated to every spin is compressed to a non-redundant integer label sequence.
           When False, spin variables are labeled (in general, but not yet implemented):
           (geometric_position, index at geometric position, quadrature, bit-precision)
           In specific, for BPSK with at most one transmitter per site, there is 1 
           spin per lattice node with a transmitter, inherits lattice label)
       F: Channel
       y: Signal
     
       See for ``spin_encoded_mimo`` for interpretation of other per-basestation parameters. 
    Returns:
       bqm: an Ising model in BinaryQuadraticModel format.
    
    Reference: 
        https://en.wikipedia.org/wiki/Cooperative_MIMO
    """
    if type(lattice) is not nx.Graph:
        lattice = _make_honeycomb(int(lattice))
    if modulation is None:
        modulation = 'BPSK'
    attenuation_matrix, ntr, ntt = lattice_to_attenuation_matrix(lattice,
                                                                 transmitters_per_node=num_transmitters_per_node,
                                                                 receivers_per_node=num_receivers_per_node,
                                                                 neighbor_root_attenuation=1)
    num_receivers, num_transmitters = attenuation_matrix.shape
    bqm = spin_encoded_mimo(modulation=modulation, y=y, F=F,
                            transmitted_symbols=transmitted_symbols, channel_noise=channel_noise, 
                            num_transmitters=num_transmitters,  num_receivers=num_receivers,
                            SNRb=SNRb, 
                            seed=seed, 
                            F_distribution=F_distribution, 
                            use_offset=use_offset,
                            attenuation_matrix=attenuation_matrix)
    # I should relabel the integer representation back to (geometric_position, index_at_position, imag/real, precision)
    # Easy case (for now) BPSK num_transmitters per site at most 1.

    if modulation == 'BPSK' and num_transmitters_per_node == 1 and integer_labeling==False: 
        rtn = {v[0]: k for k,v in ntr.items()} #Invertible mapping
        # Need to check attributes really,..
        print(rtn)
        bqm.relabel_variables({n: rtn[n] for n in bqm.variables})
    
    return bqm
