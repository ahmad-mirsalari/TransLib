#ifndef _CONF_
#define _CONF_
#include "config.h"
#define WINDOW_LEN 1000
#define DWT_LEN      1000  //input
#define DWT_LEN_OUT  1155   //output
#define LEVELS       4     //((int)ceil(log2(DWT_LEN)) //2^LEVELS=DWT_LEN if it executes all levels until max number of levels=log2(N)
#define NC      40
        

#endif 
