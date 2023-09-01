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
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express 93or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from collections import namedtuple
from functools import wraps
from itertools import product
from typing import Literal, Optional, Tuple, Union

import numpy as np

import dimod

__all__ = ['mimo', 'coordinated_multipoint', ]

mod_params = namedtuple("mod_params", ["bits_per_transmitter", 
                                       "amplitudes", 
                                       "transmitters_per_bit", 
                                       "number_of_amps", 
                                       "bits_per_symbol"])
mod_config = {
    "BPSK": mod_params(1, 1, 1, 1, 1),
    "QPSK": mod_params(2, 1, 2, 1, 1),
    "16QAM": mod_params(4, 2, 4, 2, 2),
    "64QAM": mod_params(6, 4, 6, 3, 3),
    "256QAM": mod_params(8, 8, 8, 5, 4)}

def _quadratic_form(y, F):
    """Convert :math:`O(v) = ||y - F v||^2` to sparse quadratic form.

    Constructs coefficients for the form
    :math:`O(v) = v^{\dagger} J v - 2 \Re(h^{\dagger} v) + \\text{offset}`.

    Args:
        y: Received signal as a NumPy column vector of complex or real values.

        F: Wireless channel as an :math:`i \\times j` NumPy matrix of complex
            values, where :math:`i` rows correspond to :math:`y_i` receivers
            and :math:`j` columns correspond to :math:`v_i` transmitted symbols.

    Returns:
        Three tuple of offset, as a real scalar, linear biases :math:`h`, as a dense
        real vector, and quadratic interactions, :math:`J`, as a dense real symmetric
        matrix.
    """

    if len(F.shape) != 2 or F.shape[0] != y.shape[0]:
        raise ValueError("F should have shape (n, m) for some m, n "
                         "and n should equal y.shape[1];"
                         f" given: {F.shape}, n={y.shape[1]}")

    offset = np.matmul(y.imag.T, y.imag) + np.matmul(y.real.T, y.real)
    h = -2 * np.matmul(F.T.conj(), y)    
    J = np.matmul(F.T.conj(), F)

    return offset, h, J

def _real_quadratic_form(h, J, modulation=None):
    """Separate real and imaginary parts of quadratic form.

    Unwraps objective function on complex variables as an objective function of
    concatenated real variables, first the real and then the imaginary part.

    Args:
        h: Linear biases as a dense NumPy vector.

        J: Quadratic interactions as a dense symmetric matrix.

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

    if modulation and modulation not in mod_config:
        raise ValueError(f"Unsupported modulation: {modulation}")

    if modulation != 'BPSK' and (np.iscomplex(h).any() or np.iscomplex(J).any()):
        hR = np.concatenate((h.real, h.imag), axis=0)
        JR = np.concatenate((np.concatenate((J.real, J.imag), axis=0),
                             np.concatenate((J.imag.T, J.real), axis=0)),
                             axis=1)
        return hR, JR
    return h.real, J.real

def _amplitude_modulated_quadratic_form(h, J, modulation="BPSK"):
    """Amplitude-modulate the quadratic form.

    Updates bias amplitudes for quadrature amplitude modulation.

    Args:
        h: Linear biases as a NumPy vector.

        J: Quadratic interactions as a matrix.

        modulation: Modulation. Supported values are non-quadrature modulation
            'BPSK' (the default) and quadrature modulations 'QPSK', '16QAM', 
            '64QAM', and '256QAM'.

    Returns:
        Two-tuple of amplitude-modulated linear biases, :math:`h`, as a NumPy
        vector  and amplitude-modulated quadratic interactions, :math:`J`, as
        a matrix.
    """
    if modulation not in mod_config:
        raise ValueError(f"Unsupported modulation: {modulation}")

    amps = 2 ** np.arange(mod_config[modulation].number_of_amps)
    hA = np.kron(amps[:, np.newaxis], h)
    JA = np.kron(np.kron(amps[:, np.newaxis], amps[np.newaxis, :]), J)
    return hA, JA

def _symbols_to_bits(symbols, modulation="BPSK"):
    """Convert quadrature amplitude modulated (QAM) symbols to bits.

    Args:
        symbols: Transmitted symbols as a NumPy column vector.

        modulation: Modulation. Supported values are the default non-quadrature 
            modulation, binary phase-shift keying (BPSK, or 2-QAM), and 
            quadrature modulations 'QPSK', '16QAM', '64QAM', and '256QAM'.

    Returns:
        Bits as a NumPy array.
    """
    if modulation not in mod_config:
        raise ValueError(f"Unsupported modulation: {modulation}")

    if modulation == 'BPSK':
        return symbols.copy()

    if modulation == 'QPSK':
        return np.concatenate((symbols.real, symbols.imag))

    bits_per_real_symbol = mod_config[modulation].bits_per_symbol

    # A map from integer parts to real is clearest (and sufficiently performant),
    # generalizes to Gray coding more easily as well:

    symb_to_bits = {np.sum([x * 2**xI for xI, x in enumerate(bits)]): bits
        for bits in product(*bits_per_real_symbol * [(-1, 1)])}
    bits = np.concatenate(
        [np.concatenate(([symb_to_bits[symb][prec] for symb in symbols.real.flatten()],
                         [symb_to_bits[symb][prec] for symb in symbols.imag.flatten()]))
        for prec in range(bits_per_real_symbol)])
    
    if len(symbols.shape) > 2:
        raise ValueError(f"`symbols` should be 1 or 2 dimensional but is shape {symbols.shape}")
    
    if symbols.ndim == 1:    # If symbols shaped as vector, return as vector
        bits.reshape((len(bits), ))

    return bits

def _yF_to_hJ(y, F, modulation="BPSK"):
    """Convert :math:`O(v) = ||y - F v||^2` to modulated quadratic form.

    Constructs coefficients for the form
    :math:`O(v) = v^{\dagger} J v - 2 \Re(h^{\dagger} v) + \\text{offset}`.

    Args:
        y: Received symbols as a NumPy column vector of complex or real values.

        F: Wireless channel as an :math:`i \\times j` NumPy matrix of complex
            values, where :math:`i` rows correspond to :math:`y_i` receivers
            and :math:`j` columns correspond to :math:`v_i` transmitted symbols.

        modulation: Modulation. Supported values are the default non-quadrature 
            modulation, 'BPSK', and quadrature modulations 'QPSK', '16QAM', 
            '64QAM', and '256QAM'.

    Returns:
        Three tuple of amplitude-modulated linear biases :math:`h`, as a NumPy
        vector, amplitude-modulated quadratic interactions, :math:`J`, as a
        matrix, and offset as a real scalar.
    """
    offset, h, J = _quadratic_form(y, F) # Conversion to quadratic form

    # Separate real and imaginary parts of quadratic form:
    h, J = _real_quadratic_form(h, J, modulation)

    # Amplitude-modulate the biases in the quadratic form:
    h, J = _amplitude_modulated_quadratic_form(h, J, modulation)

    return h, J, offset

def _bits_to_symbols(bits, 
                      modulation="BPSK",
                      num_transmitters=None):
    """Convert bits to modulated symbols.

    Args:
        bits: Bits as a NumPy array.

        modulation: Modulation. Supported values are the default non-quadrature 
            modulation, 'BPSK', and quadrature modulations 'QPSK', '16QAM', 
            '64QAM', and '256QAM'.

    Returns:
        Transmitted symbols as a NumPy vector.
    """
    if modulation not in mod_config:
        raise ValueError(f"Unsupported modulation: {modulation}")

    num_bits = len(bits)

    if num_transmitters is None:
        num_transmitters = num_bits // mod_config[modulation].transmitters_per_bit

    if num_transmitters == num_bits:
        symbols = bits
    else:
        num_amps, rem = divmod(len(bits), (2*num_transmitters))
        if num_amps > 64:
            raise ValueError('Complex encoding is limited to 64 bits in'
                             'real and imaginary parts; `num_transmitters` is'
                             'too small')
        if rem != 0:
            raise ValueError('number of bits must be divisible by `num_transmitters` '
                             'for modulation schemes')

        bitsR = np.reshape(bits, (num_amps, 2 * num_transmitters))
        amps = 2 ** np.arange(0, num_amps)[:, np.newaxis]

        symbols = np.sum(amps*bitsR[:, :num_transmitters], axis=0) \
            + 1j*np.sum(amps*bitsR[:, num_transmitters:], axis=0)

    return symbols

def _lattice_to_attenuation_matrix(lattice, 
                                   transmitters_per_node=1, 
                                   receivers_per_node=1, 
                                   neighbor_root_attenuation=1):
    """Generate an attenuation matrix from a given lattice.

    The attenuation matrix, a NumPy :class:`~numpy.ndarray` matrix with a row 
    for each receiver and column for each transmitter, specifies the expected 
    root-power of transmission between integer-indexed transmitters and 
    receivers.

    It sets uniform transmission of power for on-site transmitter-receiver pairs
    and uniform transmission from transmitters to receivers up to graph distance 
    1. 
    """
    # Developer note: this code should exploit sparsity, and track the label map.
    # The code could be generalized to account for asymmetric transmission patterns, 
    # or real-valued spatial structure.

    if any('num_transmitters' in lattice.nodes[n] for n in lattice.nodes) or \
       any('num_receivers' in lattice.nodes[n] for n in lattice.nodes):

        node_to_transmitters = {}   #Integer labels of transmitters at node
        node_to_receivers = {}      #Integer labels of receivers at node
        t_ind = 0
        r_ind = 0
        for n in lattice.nodes:
            num = transmitters_per_node
            if 'num_transmitters' in lattice.nodes[n]:
                num = lattice.nodes[n]['num_transmitters']
            node_to_transmitters[n] = list(range(t_ind, t_ind + num))
            t_ind = t_ind + num

            num = receivers_per_node
            if 'num_receivers' in lattice.nodes[n]:
                num = lattice.nodes[n]['num_receivers']
            node_to_receivers[n] = list(range(r_ind, r_ind + num))
            r_ind = r_ind + num

        A = np.zeros(shape=(r_ind, t_ind))
        for n0 in lattice.nodes:
            root_receivers = node_to_receivers[n0]
            for r in root_receivers:
                for t in node_to_transmitters[n0]:
                    A[r, t] = 1
                for neigh in lattice.neighbors(n0):
                    for t in node_to_transmitters[neigh]:
                        A[r, t] = neighbor_root_attenuation
    else:
        A = np.identity(lattice.number_of_nodes())
        # Uniform case:
        node_to_int = {n:idx for idx, n in enumerate(lattice.nodes())}
        for n0 in lattice.nodes:
            root = node_to_int[n0]
            for neigh in lattice.neighbors(n0):
                A[node_to_int[neigh], root] = neighbor_root_attenuation
        A = np.tile(A, (receivers_per_node, transmitters_per_node))

        node_to_receivers = {n: [node_to_int[n] + i*len(node_to_int) for 
            i in range(receivers_per_node)] for n in node_to_int}

        node_to_transmitters = {n: [node_to_int[n] + i*len(node_to_int) for 
            i in range(transmitters_per_node)] for n in node_to_int}

    return A, node_to_transmitters, node_to_receivers

def create_channel(num_receivers: int = 1, 
                   num_transmitters: int = 1,
                   F_distribution: Optional[Tuple[str, str]] = None,
                   random_state: Optional[Union[int, np.random.mtrand.RandomState]] = None,
                   attenuation_matrix: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """Create a channel model.

    Args:
        num_receivers: Number of receivers.

        num_transmitters: Number of transmitters.

        F_distribution: Distribution for the channel. Supported values are:

        *   First value: ``normal`` and ``binary``.
        *   Second value: ``real`` and ``complex``.

        random_state: Seed for a random state or a random state.

        attenuation_matrix: Root of the power associated with each 
            transmitter-receiver channel. 

    Returns:
        Two-tuple of channel and channel power, where the channel is an 
        :math:`i \times j` matrix with :math:`i` rows corresponding to the 
        receivers and :math:`j` columns to the transmitters, and channel power 
        is a number.
    """

    if num_receivers < 1 or num_transmitters < 1:
        raise ValueError('At least one receiver and one transmitter are required.')
    
    channel_power = num_transmitters

    random_state = np.random.default_rng(random_state)
    
    if F_distribution is None:
        F_distribution = ('normal', 'complex')
    elif type(F_distribution) is not tuple or len(F_distribution) != 2:
        raise ValueError('F_distribution should be a tuple of strings or None')

    if F_distribution[0] == 'normal':
        if F_distribution[1] == 'real':
            F = random_state.normal(0, 1, size=(num_receivers, num_transmitters))
        else:
            channel_power = 2 * num_transmitters
            F = random_state.normal(0, 1, size=(num_receivers, num_transmitters)) + \
                1j*random_state.normal(0, 1, size=(num_receivers, num_transmitters))
    elif F_distribution[0] == 'binary':
        if F_distribution[1] == 'real':
            F = (1 - 2*random_state.integers(2, size=(num_receivers, num_transmitters)))
        else:
            channel_power = 2*num_transmitters      #For integer precision purposes:
            F = ((1 - 2*random_state.integers(2, size=(num_receivers, num_transmitters))) +
                 1j*(1 - 2*random_state.integers(2, size=(num_receivers, num_transmitters))))

    if attenuation_matrix is not None:
        if np.iscomplex(attenuation_matrix).any():
            raise ValueError('attenuation_matrix must not have complex values')
        F = F * attenuation_matrix    #Dense format for now, this is slow.
        channel_power *= np.mean(np.sum(attenuation_matrix * attenuation_matrix,
                                        axis=0)) / num_receivers

    return F, channel_power
    
def _constellation_properties(modulation):
    """Return bits per symbol, amplitudes, and mean power for QAM constellation.

    Constellation mean power makes the standard assumption that symbols are
    sampled uniformly at random for the signal.
    """
    if modulation not in mod_config:
        raise ValueError('Unsupported modulation method')

    amps = 1 + 2*np.arange(mod_config[modulation].amplitudes)
    constellation_mean_power = 1 if modulation == 'BPSK' else 2 * np.mean(amps*amps)

    return mod_config[modulation].bits_per_transmitter, amps, constellation_mean_power

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

        amps: Amplitudes as an iterable.

        quadrature: Quadrature (True) or only phase-shift keying such as BPSK (False).

        random_state: Seed for a random state or a random state.

    Returns:

        Symbols, as a column vector of length ``num_transmitters``.

    """

    if np.iscomplex(amps).any():
        raise ValueError('Amplitudes cannot have complex values')

    if any(np.modf(amps)[0]):
        raise ValueError('Amplitudes must have integer values')

    random_state = np.random.default_rng(random_state)

    if not quadrature:
        transmitted_symbols = random_state.choice(amps, size=(num_transmitters, 1))
    else:
        transmitted_symbols = random_state.choice(amps, size=(num_transmitters, 1)) \
            + 1j*random_state.choice(amps, size=(num_transmitters, 1))

    return transmitted_symbols

def _create_signal(F, 
                   transmitted_symbols=None, 
                   channel_noise=None, 
                   SNRb=float('Inf'), 
                   modulation='BPSK', 
                   channel_power=None, 
                   random_state=None):
    """Simulate a transmission signal. 

    Generates random transmitted symbols and optionally noise, math:`y = F v + n`,
    where the channel, :math:`F`, is assumed to consist of independent and 
    identically distributed (i.i.d) elements such that 
    :math:`F\dagger*F = N_r I[N_t]*cp` where :math:`I` is the identity matrix 
    and :math:`cp` the channel power; the transmitted symbols, :math:`v` or 
    ``transmitted_symbols``, are assumed to consist of i.i.d unscaled 
    constellations elements (integer valued in real and complex parts), and 
    mean constellation power dictates a rescaling relative to 
    :math:`E[v v\dagger] = I[Nt]`.

    ``channel_noise`` is assumed, or created, to be suitably scaled: 
    :math:`N_0 I[Nt] = SNRb `.

    Args:
        F: Wireless channel as an :math:`i \times j` matrix of complex values,
            where :math:`i` rows correspond to :math:`y_i` receivers and :math:`j`
            columns correspond to :math:`v_i` transmitted symbols.

        transmitted_symbols: Transmitted symbols as a column vector.

        channel_noise: Channel noise as a column vector.

        SNRb: Signal-to-noise ratio.

        modulation: Modulation. Supported values are 'BPSK', 'QPSK', '16QAM',
            '64QAM', and '256QAM'.

        channel_power: Channel power. By default, equal to the number
            of transmitters.

        random_state: Seed for a random state or a random state.

    Returns:
        Four-tuple of received signals (``y``), transmitted symbols (``v``),
        channel noise, and random_state, where ``y`` is a column vector of length
        equal to the rows of ``F``.
    """

    num_receivers = F.shape[0]
    num_transmitters = F.shape[1]

    random_state = np.random.default_rng(random_state)

    bits_per_transmitter, amps, constellation_mean_power = _constellation_properties(modulation)

    if transmitted_symbols is not None:
        if modulation == 'BPSK' and np.iscomplex(transmitted_symbols).any():
            raise ValueError(f"BPSK transmitted signals must be real")
        if modulation != 'BPSK' and np.isreal(transmitted_symbols).any():
            raise ValueError(f"Quadrature transmitted signals must be complex")
    else:
        quadrature = modulation != 'BPSK'
        transmitted_symbols = _create_transmitted_symbols(
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
            #Assume proportional to num_transmitters; i.e., every channel 
            # component is RMSE 1 and 1 bit
            channel_power = num_transmitters

        #Eb is the same for QPSK and BPSK
        # Eb/N0 = SNRb (N0 = 2 sigma^2, the one-sided PSD ~ kB T at antenna)
        # SNRb and Eb, together imply N0
        Eb = channel_power * constellation_mean_power / bits_per_transmitter 
        N0 = Eb / SNRb
        # Noise is complex by definition, hence 1/2 power in real & complex parts
        sigma = np.sqrt(N0 / 2) 

        # Channel noise of covariance N0*I_{NR}. Noise is complex by definition, 
        # although for real channel and symbols we need only worry about 
        # real part:
        channel_noise = sigma*(random_state.normal(0, 1, size=(num_receivers, 1)) \
            + 1j*random_state.normal(0, 1, size=(num_receivers, 1)))
        if modulation == 'BPSK' and np.isreal(F).all():
            channel_noise = channel_noise.real

        y = channel_noise + np.matmul(F, transmitted_symbols)

    return y, transmitted_symbols, channel_noise, random_state

def mimo(modulation: Literal["BPSK", "QPSK", "16QAM", "64QAM", "256QAM"] = "BPSK", 
         y: Union[np.array, None] = None,
         F: Union[np.array, None] = None,
         *,
         transmitted_symbols: Union[np.array, None] = None,
         channel_noise: Union[np.array, None] = None,
         num_transmitters: int = None,
         num_receivers: int = None,
         SNRb: float = float('Inf'),
         seed: Union[None, int, np.random.RandomState] = None,
         F_distribution: Union[None, tuple] = None,
         attenuation_matrix = None) -> dimod.BinaryQuadraticModel:
    """Generate a multi-input multiple-output (MIMO) channel-decoding problem.

    In radio networks, `MIMO <https://en.wikipedia.org/wiki/MIMO>`_ is a method 
    of increasing link capacity by using multiple transmission and receiving 
    antennas to exploit multipath propagation.

    Users transmit complex-valued symbols over a random channel, :math:`F`, 
    subject to additive white Gaussian noise. Given the received signal, 
    :math:`y`, the log likelihood of a given symbol set, :math:`v`, is 
    :math:`MLE = argmin || y - F v ||_2`. When :math:`v` is encoded as 
    a linear sum of bits, the optimization problem is a binary quadratic model.
    
    Depending on its parameters, this function can model code division multiple
    access (CDMA) _[#T02, #R20], 5G communication networks _[#Prince], or 
    other problems.

    Args:
        modulation: Constellation (symbol set) users can transmit. Symbols are 
            assumed to be transmitted with equal probability. Supported values 
            are:

            * 'BPSK'

                Binary Phase Shift Keying. Transmitted symbols are :math:`+1, -1`;
                no encoding is required. A real-valued channel is assumed.

            * 'QPSK'

                Quadrature Phase Shift Keying. Transmitted symbols are 
                :math:`1+1j, 1-1j, -1+1j, -1-1j` normalized by 
                :math:`\\frac{1}{\\sqrt{2}}`. Bits are encoded as a real vector 
                concatenated with an imaginary vector.

            * '16QAM'

                Each user is assumed to select independently from 16 symbols.
                The transmitted symbol is a complex value that can be encoded 
                by two bits in the imaginary part and two bits in the real 
                part. Highest precision real and imaginary bit vectors are 
                concatenated to lower precision bit vectors.

            * '64QAM'

                A QPSK symbol set is generated and symbols are further amplitude 
                modulated by an independently and uniformly distributed random 
                amount from :math:`[1, 3]`.

            * '256QAM'

                A QPSK symbol set is generated and symbols are further amplitude 
                modulated by an independently and uniformly distributed random 
                amount from :math:`[1, 3, 5]`.

        y: Complex- or real-valued received signal, as a NumPy array. If 
            ``None``, generated from other arguments.

        F: Complex- or real-valued channel, as a NumPy array. If ``None``, 
            generated from other arguments. Note that for correct interpretation
            of SNRb, channel power should be normalized to ``num_transmitters``.

        num_transmitters: Number of users. Each user transmits one symbol per 
            frame.

        num_receivers: Number of receivers of a channel. Must be consistent 
            with the length of any provided signal, ``len(y)``.

        SNRb: Signal-to-noise ratio per bit used to generate the noisy signal 
            when ``y`` is not provided. If ``float('Inf')``, no noise is
            added. 
            
            :math:`SNR_b = E_b/N_0`, where :math:`E_b` is energy per bit, 
            and :math:`N_0` is the one-sided power-spectral density. :math:`N_0` 
            is typically :math:`k_B T` at the receiver. To convert units of 
            :math:`dB` to :math:`SNR_b` use :math:`SNRb=10^{SNR_b[decibels]/10}`.

        seed: Random seed, as an integer, or state, as a 
            :class:`numpy.random.RandomState` instance.   
            
        transmitted_symbols:
            Set of symbols transmitted. Used in combination with ``F`` to 
            generate the received signal, :math:`y`. The number of transmitted 
            symbols must be consistent with ``F``.
            
            For BPSK and QPSK modulations, statistics of the ensemble do not 
            depend on the choice: all choices are equivalent. By default, 
            symbols are chosen for all users as :math:`1` or :math:`1 + 1j`, 
            respectively. Note that for correct analysis by some solvers, applying 
            spin-reversal transforms may be necessary.
            
            For QAM modulations such as 16QAM, amplitude randomness affects 
            likelihood in a non-trivial way. By default, symbols are chosen 
            independently and identically distributed from the constellations. 

        channel_noise: Channel noise as a NumPy array of complex values. Must 
            be consistent with the number of receivers. 

        F_distribution:
            Zero-mean, variance-one distribution, in tuple form 
            ``(distribution, type)``, used to generate each element in ``F`` 
            when ``F`` is not provided . Supported values are:
            
                * ``'normal'`` or ``'binary'`` for the distribution 
                * ``'real'`` or ``'complex'`` for the type 

            For large numbers of receivers and transmitters, statistical 
            properties of the likelihood are weakly dependent on the 
            distribution. Choosing ``'binary'`` allows for integer-valued 
            Hamiltonians while ``'normal'`` is a more typical model. The channel 
            can be real or complex; in many cases this represents a superficial 
            distinction up to rescaling. For real-valued symbols (BPSK) the 
            default is ``('normal', 'real')``; otherwise, the default is 
            ``('normal', 'complex')``. 

        attenuation_matrix:
           Root of the power associated with each transmitter-receiver channel; 
           use for sparse and structured codes.

    Returns:
        Binary quadratic model defining the log-likelihood function.

    Example:

        This example generates an instance of a CDMA problem in the high-load 
        regime, near a first-order phase transition.

        >>> num_transmitters = 64
        >>> transmitters_per_receiver = 1.5
        >>> SNRb = 5
        >>> bqm = dimod.generators.mimo(modulation='BPSK', 
        ...     num_transmitters = 64, 
        ...     num_receivers = round(num_transmitters / transmitters_per_receiver), 
        ...     SNRb=SNRb, 
        ...     F_distribution = ('binary', 'real'))

    .. [#T02] T. Tanaka IEEE TRANSACTIONS ON INFORMATION THEORY, VOL. 48, NO. 11, NOVEMBER 2002

    .. [#R20] J. Raymond, N. Ndiaye, G. Rayaprolu and A. D. King, 
        "Improving performance of logical qubits by parameter tuning and topology compensation," 
        2020 IEEE International Conference on Quantum Computing and Engineering (QCE), 
        Denver, CO, USA, 2020, pp. 295-305, doi: 10.1109/QCE49297.2020.00044.

    .. [#Prince] Various (https://paws.princeton.edu/)
    """

    random_state = np.random.default_rng(seed)

    if y is not None:
        if len(y.shape) == 1:
            y = y.reshape(y.shape[0], 1)
        elif len(y.shape) != 2 or y.shape[1] != 1:
            raise ValueError(f"y should have shape (n, 1) or (n,) for some n; given: {y.shape}")

    if F is None:

        if num_transmitters:
            if num_transmitters <= 0:
                raise ValueError('Configured number of transmitters must be positive')
        elif transmitted_symbols:
            num_transmitters = len(transmitted_symbols)
        else:
           ValueError('`num_transmitters` is not specified and cannot'
                'be inferred from `F` or `transmitted_symbols` (both None)') 
           
        if num_receivers:
            if num_receivers <= 0:
                raise ValueError('Configured number of receivers must be positive')
        elif y:
            num_receivers = y.shape[0]
        elif channel_noise:
            num_receivers = channel_noise.shape[0]  
        else:
            raise ValueError('`num_receivers` is not specified and cannot'
                'be inferred from `F`, `y` or `channel_noise` (all None)')  

        F, channel_power = create_channel(num_receivers=num_receivers,
                                                num_transmitters=num_transmitters,
                                                F_distribution=F_distribution,
                                                random_state=random_state,
                                                attenuation_matrix=attenuation_matrix)
    else:
        channel_power = num_transmitters

    if y is None:
        y, _, _, _ = _create_signal(F,
                                    transmitted_symbols=transmitted_symbols,
                                    channel_noise=channel_noise,
                                    SNRb=SNRb, 
                                    modulation=modulation,
                                    channel_power=channel_power,
                                    random_state=random_state)

    h, J, offset = _yF_to_hJ(y, F, modulation)

    return dimod.BQM(h[:, 0], J, 'SPIN', offset=offset)
    
def coordinated_multipoint(lattice: 'networkx.Graph',
    modulation: Literal["BPSK", "QPSK", "16QAM", "64QAM", "256QAM"] = "BPSK", 
    y: Optional[np.array] = None,
    F: Union[np.array, None] = None,
    *,
    transmitted_symbols: Union[np.array, None] = None,
    channel_noise: Union[np.array, None] = None,
    SNRb: float = float('Inf'),
    seed: Union[None, int, np.random.RandomState] = None,
    F_distribution: Union[None, str] = None) -> dimod.BinaryQuadraticModel:
    """Generate a coordinated multi-point (CoMP) decoding problem.

    In `coordinated multipoint (CoMP) <https://en.wikipedia.org/wiki/Cooperative_MIMO>`_
    neighboring cellular base stations coordinate transmissions and jointly 
    process received signals.

    Users transmit complex-valued symbols over a random channel, :math:`F`, 
    subject to additive white Gaussian noise. Given the received signal, 
    :math:`y`, the log likelihood of a given symbol set, :math:`v`, is 
    :math:`MLE = argmin || y - F v ||_2`. When :math:`v` is encoded as 
    a linear sum of bits, the optimization problem is a binary quadratic model.

    Args:
        lattice: Geometry, as a :class:`networkx.Graph`, defining 
            the set of nearest-neighbor base stations. 
            
            Each base station has ``num_receivers`` receivers and 
            ``num_transmitters`` local transmitters, set as either attributes
            of the graph or as per-node values. Transmitters from neighboring 
            base stations are also received. 
           
        modulation: Constellation (symbol set) users can transmit. Symbols are 
            assumed to be transmitted with equal probability. Supported values 
            are:

            * 'BPSK'
                Binary Phase Shift Keying. Transmitted symbols are :math:`+1, -1`;
                no encoding is required. A real-valued channel is assumed.
            * 'QPSK'
                Quadrature Phase Shift Keying. Transmitted symbols are 
                :math:`1+1j, 1-1j, -1+1j, -1-1j` normalized by 
                :math:`\\frac{1}{\\sqrt{2}}`. Bits are encoded as a real vector 
                concatenated with an imaginary vector.
            * '16QAM'
                Each user is assumed to select independently from 16 symbols.
                The transmitted symbol is a complex value that can be encoded 
                by two bits in the imaginary part and two bits in the real 
                part. Highest precision real and imaginary bit vectors are 
                concatenated to lower precision bit vectors.
            * '64QAM'
                A QPSK symbol set is generated and symbols are further amplitude 
                modulated by an independently and uniformly distributed random 
                amount from :math:`[1, 3]`.
            * '256QAM'
                A QPSK symbol set is generated and symbols are further amplitude 
                modulated by an independently and uniformly distributed random 
                amount from :math:`[1, 3, 5]`.
 
        y:  Complex- or real-valued received signal, as a NumPy array. If 
            ``None``, generated from other arguments.

        F:  Transmission channel. Currently not supported and must be ``None``.

        transmitted_symbols: Set of symbols transmitted. Used in combination 
            with ``F`` to generate the received signal, :math:`y`. The number 
            of transmitted symbols must be consistent with ``F``.
            
            For BPSK and QPSK modulations, statistics of the ensemble do not 
            depend on the choice: all choices are equivalent. By default, 
            symbols are chosen for all users as :math:`1` or :math:`1 + 1j`, 
            respectively. Note that for correct analysis by some solvers, 
            applying spin-reversal transforms may be necessary.
            
            For QAM modulations such as 16QAM, amplitude randomness affects 
            likelihood in a non-trivial way. By default, symbols are chosen 
            independently and identically distributed from the constellations. 
        
        channel_noise: Channel noise as a complex value.

        SNRb: Signal-to-noise ratio per bit, :math:`SNRb=10^{SNR_b[decibels]/10}`, 
            used to generate the noisy signal when ``y`` is not provided. 
            If ``float('Inf')``, no noise is added.  
        
        seed: Random seed, as an integer, or state, as a 
            :class:`numpy.random.RandomState` instance.
        
        F_distribution: Zero-mean, variance-one distribution, in tuple form 
            ``(distribution, type)``, used to generate each element in ``F`` 
            when ``F`` is not provided . Supported values are:
            
            * ``'normal'`` or ``'binary'`` for the distribution 
            * ``'real'`` or ``'complex'`` for the type 
            
            For large numbers of receivers and transmitters, statistical 
            properties of the likelihood are weakly dependent on the 
            distribution. Choosing ``'binary'`` allows for integer-valued 
            Hamiltonians while ``'normal'`` is a more typical model. The channel 
            can be real or complex; in many cases this represents a superficial 
            distinction up to rescaling. For real-valued symbols (BPSK) the 
            default is ``('normal', 'real')``; otherwise, the default is 
            ``('normal', 'complex')``.
        
    Returns:
        bqm: Binary quadratic model defining the log-likelihood function.

    Example:

        Generate an instance of a CDMA problem in the high-load regime, near a
        first-order phase transition:

        .. doctest::                    # TODO: reconsider example/default graph 
            :skipif: True

            >>> import networkx as nx       
            >>> G = nx.complete_graph(4)    
            >>> nx.set_node_attributes(G, values={n:2*n+1 for n in G.nodes()}, name='num_transmitters')
            >>> nx.set_node_attributes(G, values={n:2 for n in G.nodes()}, name='num_receivers')
            >>> transmitted_symbols = np.random.choice([1, -1], 
            ...     size=(sum(nx.get_node_attributes(G, "num_transmitters").values()), 1))
            >>> bqm = dimod.generators.coordinated_multipoint(G,
            ...     modulation='BPSK', 
            ...     transmitted_symbols=transmitted_symbols,
            ...     SNRb=5,
            ...     F_distribution = ('binary', 'real'))

    """

    if not hasattr(lattice, 'edges') or not hasattr(lattice, 'nodes'): # not nx.Graph:
        raise ValueError('Lattice must be a :class:`networkx.Graph`')

    attenuation_matrix, _, _ = _lattice_to_attenuation_matrix(lattice)
    
    num_receivers, num_transmitters = attenuation_matrix.shape

    bqm = mimo(
        modulation=modulation,
        y=y,
        F=F,
        transmitted_symbols=transmitted_symbols,
        channel_noise=channel_noise,
        num_transmitters=num_transmitters,
        num_receivers=num_receivers,
        SNRb=SNRb,
        seed=seed,
        F_distribution=F_distribution,
        attenuation_matrix=attenuation_matrix)

    return bqm

# Linear-filter functions. These are not used for bit-encoding MIMO problems
# and are maintained here for user convenience 
 
def linear_filter(F, method='zero_forcing', SNRoverNt=float('Inf'), PoverNt=1):
    """Construct a linear filter for estimating transmitted signals.

    Following the conventions of MacKay\ [#Mackay]_, a filter is constructed 
    for independent and identically distributed Gaussian noise at power spectral 
    density :math:`N_0`:

    :math:`N_0 I[N_r] = E[n n^{\dagger}]`

    For independent and identically distributed (i.i.d), zero mean, transmitted symbols,

    :math:`P_c I[N_t] = E[v v^{\dagger}]`

    where :math:`P_{c}`, the constellation's mean power, is equal to 
    :math:`(1, 2, 10, 42)N_t` for BPSK, QPSK, 16QAM, 64QAM respectively.

    For an i.i.d channel,

    :math:`N_r N_t = E_F[T_r[F F^{\dagger}]] \qquad \Rightarrow \qquad E[||F_{\mu, i}||^2] = 1`

    Symbols are assumed to be normalized:

    :math:`\\frac{SNR}{N_t} = \\frac{P_c}{N_0}`

    :math:`SNR_b = \\frac{SNR}{N_t B_c}`

    where :math:`B_c` is bit per symbol, equal to  :math:`(1, 2, 4, 8)`
    for BPSK, QPSK, 16QAM, 64QAM respectively.

    Typical use case: set :math:`\\frac{SNR}{N_t} = SNR_b`.

    .. [#Mackay] Matthew R. McKay, Iain B. Collings, Antonia M. Tulino.
        "Achievable sum rate of MIMO MMSE receivers: A general analytic framework"
        IEEE Transactions on Information Theory, February 2010
        arXiv:0903.0666 [cs.IT]

    Reference:

        https://www.youtube.com/watch?v=U3qjVgX2poM
    """
    if method not in ['zero_forcing', 'matched_filter', 'MMSE']:
        raise ValueError('Unsupported filtering method')

    if method == 'zero_forcing':
        # Moore-Penrose pseudo inverse
        return np.linalg.pinv(F)

    Nr, Nt = F.shape

    if method == 'matched_filter':  # F = root(Nt/P) Fcompconj
        return F.conj().T / np.sqrt(PoverNt)

    # method == 'MMSE':
    return np.matmul(
        F.conj().T,
        np.linalg.pinv(np.matmul(F, F.conj().T) + np.identity(Nr)/SNRoverNt)
            ) / np.sqrt(PoverNt)

def filter_marginal_estimator(x: np.array, modulation: str):
    """Map filter output to valid symbols.

    Takes the continuous filter output and maps each estimated symbol to 
    the nearest valid constellation value.
    """
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
        return x_R
