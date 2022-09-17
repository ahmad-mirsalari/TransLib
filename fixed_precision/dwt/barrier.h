#ifndef _BARRIER_
#define _BARRIER_


#if defined(SW_BARRIER)
#define BARRIER() sw_barrier();
#elif defined(TAS_BARRIER)
#define BARRIER() tas_barrier();
#else
#define BARRIER() pi_team_barrier();
#endif

#endif
