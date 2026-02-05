
#ifdef STATS

#define INIT_STATS()  rt_perf_t perf;

#define PRE_START_STATS()  \
  unsigned long _cycles = 0; \
  unsigned long _instr = 0; \
  unsigned long _active = 0; \
  unsigned long _ldext = 0; \
  unsigned long _tcdmcont = 0; \
  unsigned long _ldstall = 0; \
  unsigned long _imiss = 0; \
  for(int _k=0; _k<HOTTING+6*REPEAT; _k++) { \
    if (_k >= HOTTING) \
    { \
    	rt_perf_init(&perf); \
      if((_k - HOTTING) %6 == 0) rt_perf_conf(&perf,(1<<RT_PERF_CYCLES) | (1<<RT_PERF_INSTR) ); \
    	if((_k - HOTTING) %6 == 1) rt_perf_conf(&perf,(1<<RT_PERF_ACTIVE_CYCLES) ); \
    	if((_k - HOTTING) %6 == 2) rt_perf_conf(&perf,(1<<RT_PERF_LD_EXT) ); \
      if((_k - HOTTING) %6 == 3) rt_perf_conf(&perf,(1<<RT_PERF_TCDM_CONT) ); \
      if((_k - HOTTING) %6 == 4) rt_perf_conf(&perf,(1<<RT_PERF_LD_STALL) ); \
      if((_k - HOTTING) %6 == 5) rt_perf_conf(&perf,(1<<RT_PERF_IMISS) ); \
    }

#define START_STATS()  \
    if (_k >= HOTTING) \
    { \
    	rt_perf_reset(&perf); \
      rt_perf_start(&perf); \
    }

#define STOP_STATS() \
   if (_k >= HOTTING) \
    { \
      int id = pi_core_id(); \
    	rt_perf_stop(&perf); \
      rt_perf_save(&perf); \
      if((_k - HOTTING) %6 == 0) { \
        _cycles   += rt_perf_read (RT_PERF_CYCLES); \
        _instr    += rt_perf_read (RT_PERF_INSTR); \
      } \
    	if((_k - HOTTING) %6 == 1) _active   += rt_perf_read (RT_PERF_ACTIVE_CYCLES); \
      if((_k - HOTTING) %6 == 2) _ldext    += rt_perf_read (RT_PERF_LD_EXT); \
    	if((_k - HOTTING) %6 == 3) _tcdmcont += rt_perf_read (RT_PERF_TCDM_CONT); \
    	if((_k - HOTTING) %6 == 4) _ldstall  += rt_perf_read (RT_PERF_LD_STALL); \
    	if((_k - HOTTING) %6 == 5) _imiss    += rt_perf_read (RT_PERF_IMISS); \
    } \
  } \
  printf("[%d] cycles = %lu\n", pi_core_id(), _cycles/REPEAT); \
  printf("[%d] instr = %lu\n", pi_core_id(), _instr/REPEAT); \
	printf("[%d] active cycles = %lu\n", pi_core_id(), _active/REPEAT); \
  printf("[%d] ext load = %lu\n", pi_core_id(), _ldext/REPEAT); \
	printf("[%d] TCDM cont = %lu\n", pi_core_id(), _tcdmcont/REPEAT); \
	printf("[%d] ld stall = %lu\n", pi_core_id(), _ldstall/REPEAT); \
	printf("[%d] imiss = %lu\n", pi_core_id(), _imiss/REPEAT);

#else

#define INIT_STATS()
#define PRE_START_STATS()
#define START_STATS()
#define STOP_STATS()

#endif


#ifdef PARALLELIZABLE_STATS

  #define DECLARE_PAR_STATS()  unsigned long par_stats = 0;

  #define INIT_PAR_STATS()  rt_perf_t perf;

  #define START_PAR_STATS()  \
    rt_perf_init(&perf); \
    rt_perf_conf(&perf,(1<<RT_PERF_CYCLES) | (1<<RT_PERF_INSTR) ); \
    rt_perf_reset(&perf); \
    rt_perf_start(&perf);

  #ifdef BPAR
  #define STOP_PAR_STATS()  \
    rt_perf_stop(&perf); \
    rt_perf_save(&perf); \
    extern unsigned long par_stats; \
    par_stats +=  ( rt_perf_read (RT_PERF_INSTR)-6)/MIN(BPAR(iL), PAR_FACTOR);
  #else
  #define STOP_PAR_STATS()  \
    rt_perf_stop(&perf); \
    rt_perf_save(&perf); \
    extern unsigned long par_stats; \
    par_stats += rt_perf_read (RT_PERF_INSTR)-6;
  #endif

  #define PRINT_PAR_STATS() printf("[%d] par stats = %d\n", pi_core_id(), par_stats);

#else

  #define DECLARE_PAR_STATS()
  #define INIT_PAR_STATS()
  #define START_PAR_STATS()
  #define STOP_PAR_STATS()
  #define PRINT_PAR_STATS()

#endif
