#include <stdio.h>
#include <stdlib.h>

#include "pmsis.h"

#include "stats.h"

#include "config.h"
#include "data_def.h"
#include "out_ref.h"

#if defined(__GAP9__)
unsigned int GPIOs = 89;
#define WRITE_GPIO(x) pi_gpio_pin_write(GPIOs, x)
#endif
#ifndef FABRIC
PI_L2 uint32_t perf_values[ARCHI_CLUSTER_NB_PE];
#else
PI_L2 uint32_t perf_value;
#endif

int error = 0;
// #ifndef FP8
// double __extendohfdf2(float16alt value)
// {
//     float result;
//     __asm__ __volatile__("fcvt.s.ah %0, %1" : "=f"(result) : "f"(value) :);
//     return (double)result;
// }

// double __extendhfdf2(float16 value)
// {
//     float result;
//     __asm__ __volatile__("fcvt.s.h %0, %1" : "=f"(result) : "f"(value) :);
//     return (double)result;
// }
// #endif
int check_result(OUT_TYPE *x, int r)
{
    // i_cl_team_barrier();

    float diff = 0.0f;
    int err = 0;

    for (int i = 0; i < r; i++)
    {
        diff = fabs(x[i] - check[i]);
        if (diff > THR)
        {
            err++;
#ifdef VERBOSE
            printf("Error at index %d:\t ref %f\t c code %f\t error %f\n", i, check[i], x[i], diff);
#endif
        }

#ifdef PRINT_RESULTS
        printf(" at index %d:\t ref %f\t c code %f\t error %f\n", i, check[i], x[i], diff);

#endif
    }

    if (err != 0)
        printf("TEST FAILED with %d errors!!\n", err);
    else
        printf("TEST PASSED!!\n");

    return err;
}

DATA_LOCATION OUT_TYPE newClusters[N_CLUSTERS][N_COORDS];
DATA_LOCATION int newClusterSize[N_CLUSTERS];
// DATA_LOCATION OUT_TYPE* clusters[N_CLUSTERS];
DATA_LOCATION OUT_TYPE clustersData[N_CLUSTERS * N_COORDS];
DATA_LOCATION int membership[N_OBJECTS];
DATA_LOCATION OUT_TYPE delta, old_delta = 0; /* % of objects change their clusters */
DATA_LOCATION int loop;
#if NUM_CORES > 1
DATA_LOCATION OUT_TYPE local_newClusters[NUM_CORES][N_CLUSTERS][N_COORDS];
DATA_LOCATION int local_newClusterSize[NUM_CORES][N_CLUSTERS];
DATA_LOCATION OUT_TYPE local_delta[NUM_CORES];
#endif

// typedef float16    v2f16    __attribute__ ((vector_size (4)));
// typedef FLOAT    v2f16    __attribute__ ((vector_size (4)));

/* square of Euclid distance between two multi-dimensional points            */

static __attribute__((always_inline))
OUT_TYPE
euclid_dist_2(int numdims,   /* no. dimensions */
              INP_TYPE *coord1, /* [numdims] */
              OUT_TYPE *coord2) /* [numdims] */
{
    int i;
    #ifdef HWMIXED
    float ans = 0.0f;
    #else
    OUT_TYPE ans = 0.0f;
    #endif

#ifdef VECTOR_MODE

// Vectorized with FP8
#ifdef FP8
    int vect_limit = N_COORDS & ~0x3;
    int remainder_loop = N_COORDS & 0x3;
    OUT_VTYPE temp = (OUT_VTYPE){0, 0, 0, 0};
    OUT_VTYPE temp2 = (OUT_VTYPE){0, 0, 0, 0};
    INP_VTYPE a;
    INP_VTYPE b;
    for (i = 0; i < vect_limit; i += 4)
    {

        a = *((INP_VTYPE *)&coord1[i]);
        b = *((INP_VTYPE *)&coord2[i]);
        temp2 = (a - b);
        temp += temp2 * temp2;
    }
    for (int k = N_COORDS - remainder_loop; k < N_COORDS; k++ )
    { 
        INP_TYPE a = coord1[k];
        INP_TYPE b = coord2[k];
        temp[0] += (a - b) * (a - b);
    }

    ans = temp[0] + temp[1] + temp[2] + temp[3];
// Vectorized with FP16 or FP16ALT
#else
    OUT_VTYPE temp = (OUT_VTYPE){0, 0};
    INP_VTYPE temp2 = (INP_VTYPE){0, 0};
    INP_VTYPE a;
    INP_VTYPE b;
    for (i = 0; i < (N_COORDS & 0xfffffffe); i += 2)
    {

        a = *((INP_VTYPE *)&coord1[i]);
        b = *((INP_VTYPE *)&coord2[i]);
        temp2 = (a - b);
        temp += temp2 * temp2;
    }
    if ((N_COORDS & 0x00000001))
    {
        INP_TYPE a = coord1[N_COORDS - 1];
        INP_TYPE b = coord2[N_COORDS - 1];
        temp[0] += (a - b) * (a - b);
    }

    ans = temp[0] + temp[1];
#endif
// Not vectorized
#else
    #ifdef HWMIXED
    OUT_TYPE d = 0.0f;
    OUT_TYPE d1 = 0.0f;
    #else
    OUT_TYPE d = 0.0f;
    OUT_TYPE d1 = 0.0f;
    #endif
    for (i = 0; i < (N_COORDS & 0xfffffffe); i += 2)
    {
        INP_TYPE a = coord1[i];
        INP_TYPE a1 = coord1[i + 1];
        OUT_TYPE b = coord2[i];
        OUT_TYPE b1 = coord2[i + 1];

        d = (OUT_TYPE)a - (OUT_TYPE)b;
        d1 = (OUT_TYPE)a1 - (OUT_TYPE)b1;


        #ifdef HWMIXED
        ans += (float)d * (float)d;
        ans += (float)d1 * (float)d1;
        #else
        ans += (OUT_TYPE)d * (OUT_TYPE)d;
        ans += (OUT_TYPE)d1 * (OUT_TYPE)d1;
        #endif
    }
    
    if ((N_COORDS & 0x00000001))
    {
        INP_TYPE a = coord1[N_COORDS - 1];
        OUT_TYPE b = coord2[N_COORDS - 1];

        d = (OUT_TYPE)a - (OUT_TYPE)b;

        #ifdef HWMIXED
        ans += (float)d * (float)d;
        #else
        ans += (OUT_TYPE)d * (OUT_TYPE)d;
        #endif
    }

#endif

    return (OUT_TYPE)ans;
}

static __attribute__((always_inline)) int find_nearest_cluster(int numClusters,  /* no. clusters */
                                                               int numCoords,    /* no. coordinates */
                                                               INP_TYPE *object,    /* [N_COORDS] */
                                                               OUT_TYPE **clusters) /* [N_CLUSTERS][N_COORDS] */
{
    int index, i;
    OUT_TYPE dist, min_dist;

    /* find the cluster id that has min distance to object */
    index = 0;
    min_dist = euclid_dist_2(N_COORDS, object, clusters[0]);

    // for (i=1; i<N_CLUSTERS; i++) {

    i = 1;
    do
    {

        dist = euclid_dist_2(N_COORDS, object, clusters[i]);
        /* no need square root */
        if (dist < min_dist)
        { /* find the min and its array index */
            min_dist = dist;
            index = i;
        }
        i++;
    } while (i < N_CLUSTERS);

    return (index);
}

DATA_LOCATION OUT_TYPE *clusters[N_CLUSTERS];

#pragma GCC push_options
#pragma GCC optimize("Os")

// #define MAX_ITERATIONS 8
// #define THRESHOLD 0.00010f

/* return an array of cluster centers of size [N_CLUSTERS][N_COORDS]       */
int kmeans()
{
    // volatile FLOAT* clusters[N_CLUSTERS];
    int i, j, k, index;
    int core_id = pi_core_id();
#if NUM_CORES > 1
    if (core_id == 0)
    {
#endif

        loop = 0;

        // allocate a 2D space for returning variable clusters[] (coordinates
        //   of cluster centers)
        clusters[0] = &clustersData[0];
        for (i = 1; i < N_CLUSTERS; i++)
            clusters[i] = clusters[i - 1] + N_COORDS;

        // pick first N_CLUSTERS elements of objects[] as initial cluster centers
        for (i = 0; i < N_CLUSTERS; i++)
            for (j = 0; j < N_COORDS; j++)
                clusters[i][j] = (OUT_TYPE)objects[i][j];

        // initialize membership[]
        for (i = 0; i < N_OBJECTS; i++)
            membership[i] = -1;

        // need to initialize newClusterSize and newClusters[0] to all 0
        for (i = 0; i < N_CLUSTERS; i++)
        {
            newClusterSize[i] = 0;
            for (j = 0; j < N_COORDS; j++)
                newClusters[i][j] = 0.0f;
        }

#if NUM_CORES > 1
    }
    pi_cl_team_barrier();
    const int blockSize = (N_OBJECTS + NUM_CORES - 1) / NUM_CORES;
    const int start = core_id * blockSize;
    int end = start + blockSize; //(start+blockSize > N_OBJECTS? N_OBJECTS: start+blockSize);
    if (end > N_OBJECTS)
        end = N_OBJECTS;

    const int blockSize2 = (N_CLUSTERS + NUM_CORES - 1) / NUM_CORES;
    const int start2 = core_id * blockSize2;
    int end2 = start2 + blockSize2;
    if (end2 > N_CLUSTERS)
        end2 = N_CLUSTERS;

    for (j = 0; j < N_CLUSTERS; j++)
    {
        local_newClusterSize[core_id][j] = 0;
        for (k = 0; k < N_COORDS; k++)
            local_newClusters[core_id][j][k] = 0.0f;
    }

#endif
    uint32_t cluster_id = pi_cluster_id();

// Performance measurement
#ifdef STATS
#if !defined(__GAP9__)
    INIT_STATS();
    PRE_START_STATS();
    START_STATS();
#else
    pi_perf_conf(1 << PI_PERF_ACTIVE_CYCLES); // PI_PERF_INSTR
    pi_perf_reset();
    pi_perf_start();
#endif
#endif
// Start Power mesurement
#if defined(__GAP9__)
    pi_pad_function_set(GPIOs, 1);
    pi_gpio_pin_configure(GPIOs, PI_GPIO_OUTPUT);
    pi_gpio_pin_write(GPIOs, 0);
    WRITE_GPIO(0);

    WRITE_GPIO(1);
#endif

    do
    {
#if NUM_CORES == 1
        delta = 0.0f;

        for (i = 0; i < N_OBJECTS; i++)
        {
#else
        local_delta[core_id] = 0;

        for (i = start; i < end; i++)
        {
#endif
            // find the array index of nestest cluster center
            index = find_nearest_cluster(N_CLUSTERS, N_COORDS, objects[i], clusters);

            // if membership changes, increase delta by 1
#if NUM_CORES == 1
            if (membership[i] != index)
                delta += 1.0f;
#else
            if (membership[i] != index)
                local_delta[core_id] += 1.0f;
#endif
            // assign the membership to object i
            membership[i] = index;

#if NUM_CORES == 1
            // update new cluster centers : sum of objects located within
            newClusterSize[index]++;
            for (j = 0; j < N_COORDS; j++)
            {
                newClusters[index][j] += (OUT_TYPE)objects[i][j];
                // unsigned int reg;
                // __asm__ __volatile__ ("frcsr %0" : "=r" (reg): : );
                // printf("reg = %x\n", reg);
                // __asm__ __volatile__ ("fscsr x0" : : : );
                // printf("### %d %d %x\n", (int)(newClusters[index][j]*1000), (int)(objects[i][j]*1000.0f), *((int*)&objects[i][j]));
            }
#else
            // update new cluster centers : sum of all objects located
            //   within (average will be performed later)
            local_newClusterSize[core_id][index]++;
            for (j = 0; j < N_COORDS; j++)
                local_newClusters[core_id][index][j] += (OUT_TYPE)objects[i][j];
#endif
        }

#if NUM_CORES > 1

        pi_cl_team_barrier();
        if (core_id == 0)
        {
            delta = local_delta[0];
            for (j = 1; j < NUM_CORES; j++)
                delta += local_delta[j];
        }

        // let the main thread perform the array reduction
        for (i = start2; i < end2; i++)
        {
            for (j = 0; j < NUM_CORES; j++)
            {
                newClusterSize[i] += local_newClusterSize[j][i];
                asm volatile("" : : : "memory");
                local_newClusterSize[j][i] = 0;
                for (k = 0; k < N_COORDS; k++)
                {
                    newClusters[i][k] += local_newClusters[j][i][k];
                    asm volatile("" : : : "memory");
                    local_newClusters[j][i][k] = 0.0f;
                }
            }
        }
        pi_cl_team_barrier();
        if (core_id == 0)
        {
#endif

            // average the sum and replace old cluster centers with newClusters
            for (i = 0; i < N_CLUSTERS; i++)
            {
                for (j = 0; j < N_COORDS; j++)
                {
                    if (newClusterSize[i] > 0)
                    {
                        #ifdef FP8
                                clusters[i][j] = newClusters[i][j] / newClusterSize[i];
                        #else
                                clusters[i][j] = newClusters[i][j] / newClusterSize[i];
                        #endif
                    }

                    newClusters[i][j] = 0.0f; // set back to 0
                }
                newClusterSize[i] = 0; // set back to 0
            }
            delta /= N_OBJECTS;

            loop++;

#if NUM_CORES > 1
        }
        pi_cl_team_barrier();
#endif 
} while (delta > (OUT_TYPE)THRESHOLD && loop < MAX_ITERATIONS);
#if NUM_CORES > 1
    pi_cl_team_barrier();
#endif

// End Power mesurement
#if defined(__GAP9__)
    WRITE_GPIO(0);
#endif
// Stop performance measurement
#ifdef STATS
#if !defined(__GAP9__)
    STOP_STATS();
#else
    pi_perf_stop();

#ifndef FABRIC
    perf_values[core_id] = pi_perf_read(PI_PERF_ACTIVE_CYCLES); // PI_PERF_CYCLES
#else
    perf_value = pi_perf_read(PI_PERF_ACTIVE_CYCLES); // PI_PERF_CYCLES
#endif
// uint32_t cycles = pi_perf_read(PI_PERF_CYCLES);
#endif
#endif

#ifdef CHECK
    OUT_TYPE check_vect[N_CLUSTERS * N_COORDS];
    int c = 0;
    
#ifndef FABRIC
    if (pi_core_id() == 0)
#endif
    {
        for (i = 0; i < N_CLUSTERS; i++)
        {
            for (j = 0; j < N_COORDS; j++)
            {
                check_vect[c] = clusters[i][j];
                c++;
            }
        }
        error = check_result(check_vect, N_CLUSTERS * N_COORDS);
    }
#endif

// Print the performance values if GAP9 is used
#ifdef STATS
#if defined(__GAP9__)
#ifndef FABRIC
    if (core_id == 0)
        for (uint32_t i = 0; i < ARCHI_CLUSTER_NB_PE; i++)
            printf("[%d] Perf : %d cycles\n", i, perf_values[i]);
#else
    printf("Perf : %d cycles\n", perf_value);
#endif
#endif
#endif

return error;
}

#pragma GCC pop_options
