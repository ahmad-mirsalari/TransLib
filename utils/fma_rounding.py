# fma_rounding.py  (do NOT name this mpmath.py)
from dataclasses import dataclass
from mpmath import mp

@dataclass(frozen=True)
class BinFormat:
    ebits: int
    fbits: int
    @property
    def p(self): return self.fbits + 1
    @property
    def bias(self): return (1 << (self.ebits - 1)) - 1
    @property
    def emax(self): return (1 << self.ebits) - 2 - self.bias
    @property
    def emin(self): return 1 - self.bias

# IEEE formats
FP16 = BinFormat(ebits=5, fbits=10)        # binary16
BF16 = BinFormat(ebits=8, fbits=7)         # bfloat16
FP32 = BinFormat(ebits=8, fbits=23)        # binary32

def _round_ties_even_int(x):
    return int(mp.nint(x))

def _round_to_format_scalar(x, fmt: BinFormat) -> float:
    # enough precision for correct FMA rounding (~3p+4); add margin
    mp.prec = max(3*fmt.p + 30, 120)
    x = mp.mpf(x)
    if mp.isnan(x): return float('nan')
    if mp.isinf(x): return float('-inf') if x < 0 else float('inf')
    if x == 0:      return float(x)

    s = -1 if x < 0 else 1
    ax = abs(x)

    m, e = mp.frexp(ax)   # m in [0.5, 1)
    m *= 2; e -= 1        # m in [1, 2)

    # subnormals
    if e < (1 - fmt.bias):
        k = fmt.fbits - (1 - fmt.bias)
        frac = _round_ties_even_int(ax * mp.power(2, k))
        if frac == 0:
            return -0.0 if s < 0 else 0.0
        if frac >= (1 << fmt.fbits):
            # rounded into min normal
            val = mp.power(2, 1 - fmt.bias)
        else:
            val = mp.mpf(frac) * mp.power(2, -fmt.fbits) * mp.power(2, 1 - fmt.bias)
        return float(-val if s < 0 else val)

    # normals
    sig = _round_ties_even_int(m * (1 << fmt.fbits))
    if sig == (1 << (fmt.fbits + 1)):
        sig >>= 1
        e += 1

    if e > ((1 << fmt.ebits) - 2 - fmt.bias):
        return float('-inf') if s < 0 else float('inf')

    val = (mp.one + mp.mpf(sig - (1 << fmt.fbits)) * mp.power(2, -fmt.fbits)) * mp.power(2, e)
    return float(-val if s < 0 else val)

def round_value(x: float, fmt: BinFormat) -> float:
    """Round a real number once to the target format (ties-to-even)."""
    return _round_to_format_scalar(x, fmt)

def fma_round_value(a: float, b: float, c: float, fmt: BinFormat) -> float:
    """Compute (a*b)+c exactly and round once to the target format."""
    mp.prec = max(3*fmt.p + 30, 120)
    exact = mp.mpf(a) * mp.mpf(b) + mp.mpf(c)
    return _round_to_format_scalar(exact, fmt)
