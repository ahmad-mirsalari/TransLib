
#ifdef STATS

#define INIT_STATS()  \
  rt_perf_t perf; \
  unsigned long _cycles = 0; \
  unsigned long _instr = 0; \
  unsigned long _active = 0; \
  unsigned long _ldext = 0; \
  unsigned long _tcdmcont = 0; \
  unsigned long _ldstall = 0; \
  unsigned long _imiss = 0;


#define ENTER_LOOP_STATS()  \
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
      int id = rt_core_id(); \
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
    }


#define EXIT_LOOP_STATS()  \
  } \
  printf("[%d] cycles = %lu\n", rt_core_id(), _cycles/REPEAT); \
  printf("[%d] instr = %lu\n", rt_core_id(), _instr/REPEAT); \
	printf("[%d] active cycles = %lu\n", rt_core_id(), _active/REPEAT); \
  printf("[%d] ext load = %lu\n", rt_core_id(), _ldext/REPEAT); \
	printf("[%d] TCDM cont = %lu\n", rt_core_id(), _tcdmcont/REPEAT); \
	printf("[%d] ld stall = %lu\n", rt_core_id(), _ldstall/REPEAT); \
	printf("[%d] imiss = %lu\n", rt_core_id(), _imiss/REPEAT);

#else  // ! STATS

#define INIT_STATS()
#define ENTER_LOOP_STATS()
#define START_STATS()
#define STOP_STATS()
#define EXIT_LOOP_STATS()

#endif
