import numpy as np
from ._utils import Array, _array_input

__all__ = [
    'Constant', 'MultiConstant', 'Linear', 'Parabola', 'Cubic', 'Sine',
]


class meanFunction():
    """ Base class for mean functions"""
    _parsize = 0

    def __init__(self, *pars):
        self.pars = np.array(pars, dtype=float)

    def __repr__(self):
        return "{0}({1})".format(self.__class__.__name__,
                                 ", ".join(map(str, self.pars)))

    def get_parameters(self):
        return self.pars

    @_array_input
    def set_parameters(self, p):
        msg = f'too few parameters for mean {self.__class__.__name__}'
        assert len(p) >= self.pars.size, msg
        if len(p) > self.pars.size:
            p = list(p)
            self.pars = np.array(p[:self.pars.size], dtype=float)
            for _ in range(self.pars.size):
                p.pop(0)
            return np.array(p)
        else:
            self.pars = p

    def __add__(self, b):
        return Sum(self, b)

    def __radd__(self, b):
        return self.__add__(b)

    def __mul__(self, b):
        return Product(self, b)

    def __rmul__(self, b):
        return self.__mul__(b)


class Sum(meanFunction):
    """ Sum of two mean functions """
    def __init__(self, m1, m2):
        self.m1, self.m2 = m1, m2
        if m1.__class__ == m2.__class__:
            # if they are the same class, number the parameter names
            param_names = []
            for p in m1._param_names:
                param_names.append(f'{p}1')
            for p in m2._param_names:
                param_names.append(f'{p}2')
            self._param_names = tuple(param_names)
        else:
            self._param_names = tuple(
                list(m1._param_names) + list(m2._param_names))
        self._parsize = m1._parsize + m2._parsize
        self.pars = np.r_[self.m1.pars, self.m2.pars]

    @_array_input
    def set_parameters(self, p):
        msg = f'too few parameters for mean {self.__class__.__name__}'
        assert len(p) >= self.pars.size, msg
        if len(p) > self.pars.size:
            self.pars = np.array(p[:self.pars.size], dtype=float)
            p = self.m1.set_parameters(p)
            p = self.m2.set_parameters(p)
            return p
        else:
            self.pars = p
            p = self.m1.set_parameters(p)
            p = self.m2.set_parameters(p)

    def __repr__(self):
        return "{0} + {1}".format(self.m1, self.m2)

    @_array_input
    def __call__(self, t):
        return self.m1(t) + self.m2(t)


class Product(meanFunction):
    """ Product of two mean functions """
    def __init__(self, m1, m2):
        self.m1, self.m2 = m1, m2
        self._param_names = tuple(
            list(m1._param_names) + list(m2._param_names))
        self._parsize = m1._parsize + m2._parsize
        self.pars = np.r_[self.m1.pars, self.m2.pars]

    @_array_input
    def set_parameters(self, p):
        msg = f'too few parameters for mean {self.__class__.__name__}'
        assert len(p) >= self.pars.size, msg
        if len(p) > self.pars.size:
            self.pars = np.array(p[:self.pars.size], dtype=float)
            p = self.m1.set_parameters(p)
            p = self.m2.set_parameters(p)
            return p
        else:
            self.pars = p
            p = self.m1.set_parameters(p)
            p = self.m2.set_parameters(p)

    def __repr__(self):
        return "{0} * {1}".format(self.m1, self.m2)

    @_array_input
    def __call__(self, t):
        return self.m1(t) * self.m2(t)


class Constant(meanFunction):
    """
    A constant mean function

    Args:
        c: The constant value of the mean function
    """
    _param_names = 'c',
    _parsize = 1

    def __init__(self, c: float):
        super(Constant, self).__init__(c)

    @_array_input
    def __call__(self, t):
        return np.full(t.shape, self.pars[0])


class MultiConstant(meanFunction):
    """
    Contant mean function for multiple instruments

    Args:
        offsets: Values of the between-instrument offsets and the average
                 value of the last instrument: [off_1, off_2, ..., avg_n]  
                 Offsets are relative to the last instrument.
        obsid: Indices of observations corresponding to each instrument. These
               should be one-based: [1, 1, ..., 2, 2, 2, ..., 3]
        time: Observed times. Should be the same size as `obsid`.
    """
    _parsize = 0

    def __init__(self, offsets: np.ndarray, obsid: np.ndarray, time: np.ndarray):
        self.obsid = obsid
        self.time = time
        self._parsize = (np.ediff1d(obsid) == 1).sum() + 1
        self.ii = obsid.astype(int) - 1

        if isinstance(offsets, float):
            offsets = [offsets]

        msg = 'wrong number of parameters, '
        msg += f'expected {self._parsize} got {len(offsets)}'
        assert len(offsets) == self._parsize, msg

        super().__init__(*offsets)
        self._param_names = [f'off{i}' for i in range(1, self._parsize)]
        self._param_names += ['mean']

    def time_bins(self):
        _1 = self.time[np.ediff1d(self.obsid, 0, None) != 0]
        _2 = self.time[np.ediff1d(self.obsid, None, 0) != 0]
        offset_times = np.mean((_1, _2), axis=0)
        return np.sort(np.r_[self.time[0], offset_times])

    @_array_input
    def __call__(self, t):
        offsets, c = self.pars[:-1], self.pars[-1]
        offsets = np.pad(offsets, (0, 1))

        if t.size == self.time.size:
            ii = self.ii
        else:
            time_bins = self.time_bins()
            ii = np.digitize(t, time_bins) - 1

        m = np.full_like(t, c) + np.take(offsets, ii)
        return m


class Linear(meanFunction):
    """
    A linear mean function, using the mean time as reference  
      `m(t) = slope * (t - mean(t)) + intercept`

    Args:
        slope: The slope of the linear function
        intercept: The intercept, using mean(t) as reference
    """
    _param_names = ('slope', 'intercept')
    _parsize = 2

    def __init__(self, slope: float, intercept: float):
        super(Linear, self).__init__(slope, intercept)

    @_array_input
    def __call__(self, t):
        tmean = t.mean()
        return self.pars[0] * (t - tmean) + self.pars[1]


class Parabola(meanFunction):
    """
    A 2nd degree polynomial mean function  
      `m(t) = quad * t² + slope * t + intercept`

    Args:
        quad: The quadratic term
        slope: The linear term
        intercept: The intercept
    """
    _param_names = ('slope', 'intercept', 'quadratic')
    _parsize = 3

    def __init__(self, quad: float, slope: float, intercept: float):
        super(Parabola, self).__init__(quad, slope, intercept)

    @_array_input
    def __call__(self, t):
        return np.polyval(self.pars, t)


class Cubic(meanFunction):
    """
    A 3rd degree polynomial mean function  
      `m(t) = cub * t³ + quad * t² + slope * t + intercept`

    Args:
        cub: The cubic term
        quad: The quadratic term
        slope: The linear term
        intercept: The intercept
    """
    _param_names = ('cub', 'quad', 'slope', 'intercept')
    _parsize = 4

    def __init__(self, cub: float, quad: float, slope: float, intercept: float):
        super(Cubic, self).__init__(cub, quad, slope, intercept)

    @_array_input
    def __call__(self, t):
        return np.polyval(self.pars, t)


class Sine(meanFunction):
    """
    A sinusoidal mean function  
      `m(t) = amplitude * sin(2*pi*t/P + phase)`

    Args:
        amplitude: The amplitude of the sinusoidal function
        period: The period of the sinusoidal function
        phase: The phase of the sinusoidal function
    """
    _param_names = ('amplitude', 'period', 'phase')
    _parsize = 3

    def __init__(self, amplitude: float, period: float, phase: float):
        super(Sine, self).__init__(amplitude, period, phase)

    @_array_input
    def __call__(self, t):
        A, P, φ = self.pars
        return A * np.sin((2 * np.pi * t / P) + φ)


# class Keplerian(meanFunction):
#     """ 
#     Keplerian function

#     Error: The Keplerian mean function is not yet implemented
         
#     """
#     _param_names = ('P', 'K', 'e', 'w', 'Tp')
#     _parsize = 5

#     def __init__(self, P, K, e, w, Tp):
#         raise NotImplementedError
#         super(Keplerian, self).__init__(P, K, e, w, Tp)

#     @_array_input
#     def __call__(self, t):
#         P, K, e, w, Tp = self.pars
#         return keplerian(t, P, K, e, w, Tp, 0.0)


# class CubicSun(meanFunction):
#     """
#     A 3rd degree mean function
#     m(t) = (t - xshift)**(-3) + yshift
#     """
#     _parsize = 4

#     def __init__(self, xshift, yshift):
#         super(Cubic, self).__init__(xshift, yshift)
#         self.xshift = xshift
#         self.yshift = yshift

#     @_array_input
#     def __call__(self, t):
#         return (t - self.xshift)**(-3) + self.yshift

