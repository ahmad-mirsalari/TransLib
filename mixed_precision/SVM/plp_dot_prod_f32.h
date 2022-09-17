#include <stdint.h>
#include "defines.h"

volatile void plp_dot_prod_f32(
                         const INP_TYPE * __restrict__ pSrcA,
                         const INP_TYPE * __restrict__ pSrcB,
                         uint32_t blockSize,
                         OUT_TYPE * __restrict__ pRes);