#ifndef _STATS_H
#define _STATS_H

#define HOTTING 2
#define REPEAT  5

#ifdef BOARD

#include "stats_board.h"

#else

#ifdef STATS

#define INIT_STATS()

#define PRE_START_STATS()  \
    unsigned long _cycles = 0; \
    unsigned long _instr = 0; \
    unsigned long _active = 0; \
    unsigned long _ldext = 0; \
    unsigned long _tcdmcont = 0; \
    unsigned long _ldstall = 0; \
    unsigned long _imiss = 0; \
    unsigned long _apu_cont = 0; \
    unsigned long _apu_dep = 0; \
    unsigned long _apu_type = 0; \
    unsigned long _apu_wb = 0; \
    for(int _k=0; _k<HOTTING+REPEAT; _k++) { \
      pi_perf_conf((1<<PI_PERF_CYCLES) | (1<<PI_PERF_INSTR) | (1<<PI_PERF_ACTIVE_CYCLES) | (1<<PI_PERF_LD_EXT) | (1<<PI_PERF_TCDM_CONT) | (1<<PI_PERF_LD_STALL) | (1<<PI_PERF_IMISS) | (1<<0x11)| (1<<0x12)| (1<<0x13)| (1<<0x14));


#define START_STATS()  \
    pi_perf_reset(); \
    pi_perf_start();

#define STOP_STATS() \
   pi_perf_stop(); \
   if (_k >= HOTTING) \
    { \
      _cycles   += pi_perf_read (PI_PERF_CYCLES); \
      _instr    += pi_perf_read (PI_PERF_INSTR); \
      _active   += pi_perf_read (PI_PERF_ACTIVE_CYCLES); \
      _ldext    += pi_perf_read (PI_PERF_LD_EXT); \
      _tcdmcont += pi_perf_read (PI_PERF_TCDM_CONT); \
      _ldstall  += pi_perf_read (PI_PERF_LD_STALL); \
      _imiss    += pi_perf_read (PI_PERF_IMISS); \
      _apu_type    += __SPRREAD (0x791); \
      _apu_cont += pi_perf_read (0x12); \
      _apu_dep  += pi_perf_read (0x13); \
      _apu_wb    += pi_perf_read (0x14); \
    } \
   if (_k == HOTTING+REPEAT-1) \
    { \
      int id = pi_core_id(); \
      printf("[%d] cycles = %lu\n", id, _cycles/REPEAT); \
      printf("[%d] instr = %lu\n", id, _instr/REPEAT); \
      printf("[%d] active cycles = %lu\n", id, _active/REPEAT); \
      printf("[%d] ext load = %lu\n", id, _ldext/REPEAT); \
      printf("[%d] TCDM cont = %lu\n", id, _tcdmcont/REPEAT); \
      printf("[%d] ld stall = %lu\n", id, _ldstall/REPEAT); \
      printf("[%d] imiss = %lu\n", id, _imiss/REPEAT); \
      printf("[%d] _apu_type  = %lu\n", id, _apu_type/REPEAT); \
      printf("[%d] _apu_cont  = %lu\n", id, _apu_cont/REPEAT); \
      printf("[%d] _apu_dep  = %lu\n", id, _apu_dep/REPEAT); \
      printf("[%d] _apu_wb  = %lu\n", id, _apu_wb/REPEAT); \
    } \
  }

#else // STATS

#define INIT_STATS()
#define PRE_START_STATS()
#define START_STATS()
#define STOP_STATS()

#endif  // STATS


#ifdef PARALLELIZABLE_STATS

  #define DECLARE_PAR_STATS()  unsigned long par_stats = 0;

  #define INIT_PAR_STATS()  pi_perf_t perf;

  #define START_PAR_STATS()  \
    pi_perf_init(); \
    pi_perf_conf(,(1<<PI_PERF_CYCLES) | (1<<PI_PERF_INSTR) ); \
    pi_perf_reset(); \
    pi_perf_start();

  #ifdef BPAR
  #define STOP_PAR_STATS()  \
    pi_perf_stop(); \
    pi_perf_save(); \
    extern unsigned long par_stats; \
    par_stats +=  ( pi_perf_read (PI_PERF_INSTR)-6)/MIN(BPAR(iL), PAR_FACTOR);
  #else
  #define STOP_PAR_STATS()  \
    pi_perf_stop(); \
    pi_perf_save(); \
    extern unsigned long par_stats; \
    par_stats += pi_perf_read (PI_PERF_INSTR)-6;
  #endif

  #define PRINT_PAR_STATS() printf("[%d] par stats = %d\n", pi_core_id(), par_stats);

#else // PARALLELIZABLE_STATS

  #define DECLARE_PAR_STATS()
  #define INIT_PAR_STATS()
  #define START_PAR_STATS()
  #define STOP_PAR_STATS()
  #define PRINT_PAR_STATS()

#endif // PARALLELIZABLE_STATS

#endif // WOLFE

#endif
