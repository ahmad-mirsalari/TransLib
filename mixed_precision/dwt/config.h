#ifndef _CONF_
#define _CONF_
#include "config.h"
#define WINDOW_LEN 128
#define DWT_LEN 128  // input
#define DWT_LEN_OUT 154   // output
#define LEVELS 4     // ((int)ceil(log2(DWT_LEN))) // 2^LEVELS = DWT_LEN if executed fully until max levels = log2(N)
#define NC 8

#endif
