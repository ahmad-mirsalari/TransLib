#include <alloca.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "rt/rt_api.h"
#include <stdio.h>
#include <math.h>

void normalize(float *x, int length, float max_val)
{
    for(i = 0; i < length; i++)
        y[i] = (x[i] - mean_val) / max_val; 
}