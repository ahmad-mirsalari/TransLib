#include "barrier.h"

#ifdef SW_BARRIER

RT_L1_DATA unsigned int barrier_lock = 0;
RT_L1_DATA int counter = 0;
RT_L1_DATA volatile unsigned int core_docked[8];

void sw_barrier()
{
  while (rt_tas_lock_32((unsigned int)&barrier_lock) == -1L);
  if (counter == 0) for(int i=0; i<NUM_CORES; i++) core_docked[i] = 0;
  counter++;

  if (counter == NUM_CORES) {
     counter = 0;
     for(int i=0; i<NUM_CORES; i++) core_docked[i] = 1;
     rt_tas_unlock_32((unsigned int)&barrier_lock, 0);
  }
  else  {
    rt_tas_unlock_32((unsigned int)&barrier_lock, 0);
    while(!core_docked[rt_core_id()]);
  }
}

#endif


#ifdef TAS_BARRIER

RT_L1_DATA volatile unsigned int __rt_barrier_status = 0;
RT_L1_DATA unsigned int __rt_barrier_wait_mask;

void init_tas_barrier()
 {
  __rt_barrier_wait_mask = (1<<NUM_CORES) - 1;
}

void tas_barrier()
{
  int core_id = rt_core_id();
  unsigned int status;
  while ((status = rt_tas_lock_32((unsigned int)&__rt_barrier_status)) == -1UL)
  {
    eu_evt_maskWaitAndClr(1<<RT_CL_SYNC_EVENT);
  }
  status |= 1<<core_id;
  if (status == __rt_barrier_wait_mask)
  {
    status = 0;
  }
  rt_tas_unlock_32((unsigned int)&__rt_barrier_status, status);
  eu_evt_trig(eu_evt_trig_addr(RT_CL_SYNC_EVENT), 0);

  while ((__rt_barrier_status >> core_id) & 1)
  {
    eu_evt_maskWaitAndClr(1<<RT_CL_SYNC_EVENT);
  }
}

#endif
