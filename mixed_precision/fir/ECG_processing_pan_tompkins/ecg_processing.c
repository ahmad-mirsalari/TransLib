/**
 * ------------------------------------------------------------------------------*
 * File: ecg_processing.c                                                        *
 *                                                                               *
 * Author: Benedetta Mazzoni <b.mazzoni@unibo.it>                                *
 *         Giuseppe Tagliavini <g.tagliavini@unibo.it>                           *
 *                                                                               *
 * ------------------------------------------------------------------------------*
 * ---------------------------------- HISTORY ---------------------------------- *
 *    date   |    author    |                     description                    *
 * ----------| -------------| ---------------------------------------------------*
 * 2021/11/11| Mazzoni B.   | - Added normalization step                         *
 *           | Tagliavini G.| - Fixed output buffer size for each filters and    *
 *           |              | function application.                              *
 *           |              | - Added performance counter                        *
 * 2021/12/20| Mazzoni B.   | - Improved R peaks detection runtime considering   *
 *           | Tagliavini G.| only the integrated signal.                        *
 *           |              | - Computed Heart Rate (beats per minute)           *
 *           |              | from the RR average value.   
 *
 * 2025/05/20| Mirsalari A. | - Added support for FP8 and FP16 data types.       *
 *
 * ------------------------------------------------------------------------------*
 ** Description                                                                  *
 *                                                                               *
 * To compute the HR, we adopt a signal processing pipeline based on the Pan and *
 * Tompkins technique. This methodology adopts a dual-threshold technique to     *
 * detect the R peaks and includes multiple pre-processing signal steps required *
 * to improve the signal analysis.                                               *
 * The signal processing pipeline includes a set of pre-processing digital       *
 * filters followed by the computation of R-peaks. The original implementation   *
 * considers a sampling rate of 200 Hz.                                          *
 * The code supports both buffered and the data-streaming simulation with        *
 * configurable parameters for the sampling frequency.                           *
 * In the case of buffered execution, the input buffer size is selected to       *
 * contain at least 1.66 times an R-R interval, considering that the maximum     *
 * physiological beats per minute are 60 or 80 (max 1.66 beats per second).      *
 * The code includes buffers for the results of the intermediate filters. These  *
 * buffers have the exact window size for the corresponding filter and are       *
 * implemented as circular buffers to reduce memory consumption.                 *
 * Buffered execution can be used to execute the algorithm on pre-recorded ECG   *
 * datasets, while the streaming variant is more efficient for real-time data    *
 * acquisition.                                                                  *
 *-------------------------------------------------------------------------------*
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <alloca.h>
#include <limits.h>
#include <stdio.h>

#include "pmsis.h"
#include "stats.h"

#include "ecg_processing.h"
#include "kernels.def" // filters coefficients
#include "data.h"      // change FS, max_value, and ECG_LEN parameters in ecg_processing.h (cfr. MATLAB)
#include "ref.h"
// #include "ecg_VarName1.h"   // check FS, max_value, and ECG_LEN parameters in ecg_processing.h (cfr. MATLAB)

//********//
// Per eseguire fare il source: source /home/mazzoni/Pulp/sourceme.sh
// To execute only on the FC: make clean all run FABRIC=1
//********//

/* #pragma GCC push_options
#if (NUM_CORES>1)
#pragma GCC optimize ("no-tree-ch")
#pragma GCC optimize ("no-tree-loop-im")
#endif */


#if defined(__GAP9__)
unsigned int GPIOs = 89;
#define WRITE_GPIO(x) pi_gpio_pin_write(GPIOs,x)
#endif
#ifndef FABRIC
PI_L2 uint32_t perf_values[ARCHI_CLUSTER_NB_PE];
#else
PI_L2 uint32_t perf_value;
#endif

// The signal array is where the most recent samples are kept.
// The other arrays are the outputs of each filtering module
dataType signal_data[BUFFER_SIZE];
OUT_TYPE lowpass_data[BUFFER_SIZE], highpass_data[BUFFER_SIZE], derivative_data[BUFFER_SIZE], squaring_data[BUFFER_SIZE], integrated_data[BUFFER_SIZE];

// Position of the last peaks
int R_loc[N_AVG];
// int R_loc[N];

// Algorithm status (global)
qrsdataType peaki = 0, spki = 0, npki = 0, threshold1 = 0, threshold2 = 0;
static __uint32_t previous_peak = 0;
static __uint32_t peak_counter = 0;
static int searchback_end = 0;
static int initialized = 0;
static int peak_processing = 0;
#ifdef VECTORIAL

#ifdef FP16_VECTORIATION // TODO: when input datatype is fp16
INP_VTYPE *signal_data_v = (INP_VTYPE *)&signal_data[0];
INP_VTYPE SV;
#else  //  ***************** ATTENTION: IN FP8, WE USE FP16 FOR THE SIGNAL DATA ****************
v2f16 *signal_data_v = (v2f16 *)&signal_data[0];
v2f16 SV;
#endif
INP_VTYPE AV;
INP_VTYPE *lowpass_data_v = (INP_VTYPE *)&lowpass_data[0];
INP_VTYPE *highpass_data_v = (INP_VTYPE *)&highpass_data[0];
INP_VTYPE *derivative_data_v = (INP_VTYPE *)&derivative_data[0];
INP_VTYPE *integrated_data_v = (INP_VTYPE *)&integrated_data[0];
#endif

// uint32_t readECGSamples_total;
// int16_t input_data[33];

OUT_TYPE single_convolution(OUT_TYPE *x, int n, FIL_TYPE *h, int nc);
OUT_TYPE single_convolution_mix(dataType *x, int n, FIL_TYPE *h, int nc);

int done = 0;

// __attribute__ ((noinline))
//__attribute__ ((always_inline))

hrdataType ecg_processing(dataType *input_data, int sample)
{
    hrdataType hr = 0;

    // Signal status variables
    int current_signal, current_lo, current_hi, current_der, current_int;

    // Detection parameters
    bool_enum is_qrs = false;
    const dataType min_rr_width = (dataType)(.2f) * FS;
    const int max_rr_width = BUFFER_SIZE; // BUFFER_SIZE = 1.6*FS
    int dist_previous_peak = 0;
    int dist_searchback_end = 0;

    // Perform a shifting inside the data buffers
    if (initialized || sample >= NC_Lo)
    {
#ifdef FP16_VECTORIATION 

        for (int i = 0; i < (NC_Lo) / 2; i +=  1)
        {
            SV = *((INP_VTYPE *)&signal_data[ 2 * i + 1]);
            signal_data_v[i] = (INP_VTYPE)SV;//signal_data_v[i + 1];

        }
#elif defined(FP8_VECTORIATION) //  ***************** ATTENTION: IN FP8, WE USE FP16 FOR THE SIGNAL DATA *********************
        
        // for (int i = 0; i < (NC_Lo) / 2; i +=  1)
        // {
        //     SV = *((v2f16 *)&signal_data[ 2 * i + 1]);
        //     signal_data_v[i] = (v2f16)SV;//signal_data_v[i + 1];

        // }

        // if you are using pure FP8, change v2f16 to float8 in ecg_processing.h and use:
        for (int i = 0; i < (NC_Lo) / 4; i +=  1)
        {
            SV = *((v2f16 *)&signal_data[ 4 * i + 1]);
            signal_data_v[i] = (v2f16)SV;//signal_data_v[i + 1];

        }
#else
        for (int i = 0; i < NC_Lo - 1; i++)
        {
            signal_data[i] = signal_data[i + 1];
        }
#endif
        current_signal = NC_Lo - 1;
    }
    else
    {
        current_signal = sample;
    }

    if (initialized || sample >= NC_Hi)
    {

#ifdef FP16_VECTORIATION 

        for (int i = 0; i < (NC_Hi) / 2; i += 1)
        {
            AV = *((INP_VTYPE *)&lowpass_data[ 2 * i + 1]);

            lowpass_data_v[i] = (INP_VTYPE)AV;//lowpass_data_v[i + 1];
        }
#elif defined(FP8_VECTORIATION) 

        for (int i = 0; i < (NC_Hi) / 4; i += 1)
        {
            AV = *((INP_VTYPE *)&lowpass_data[ 4 * i + 1]);
            lowpass_data_v[i] = (INP_VTYPE)AV;//lowpass_data_v[i + 1];
        }

#else
        for (int i = 0; i < NC_Hi - 1; i++)
        {
            lowpass_data[i] = lowpass_data[i + 1];
        }
#endif

        current_lo = NC_Hi - 1;
    }
    else
    {
        current_lo = sample;
    }

    if (initialized || sample >= NC_Der)
    {
#ifdef FP16_VECTORIATION 


        for (int i = 0; i < (NC_Der) / 2; i += 1)
        {
            AV = *((INP_VTYPE *)&highpass_data[ 2 * i + 1]);
            highpass_data_v[i] = (INP_VTYPE)AV;//highpass_data_v[i + 1];
        }
#elif defined(FP8_VECTORIATION)
        for (int i = 0; i < (NC_Der) / 4; i += 1)
        {
            AV = *((INP_VTYPE *)&highpass_data[ 4 * i + 1]);
            highpass_data_v[i] = (INP_VTYPE)AV;//highpass_data_v[i + 1];
        }
#else
        for (int i = 0; i < NC_Der - 1; i++)
        {
            highpass_data[i] = highpass_data[i + 1];
        }
#endif
        current_hi = NC_Der - 1;
    }
    else
    {
        current_hi = sample;
    }

    if (initialized || sample >= NC_Int)
    {
#ifdef FP16_VECTORIATION 

        for (int i = 0; i < (NC_Int) / 2; i += 1)
        {
            AV = *((INP_VTYPE *)&derivative_data[ 2 * i + 1]);
            derivative_data_v[i] = (INP_VTYPE)AV;//derivative_data_v[i + 1];
        }
#elif defined(FP8_VECTORIATION)
        for (int i = 0; i < (NC_Int) / 4; i += 1)
        {
            AV = *((INP_VTYPE *)&derivative_data[ 4 * i + 1]);
            derivative_data_v[i] = (INP_VTYPE)AV;//derivative_data_v[i + 1];
        }
#else
        for (int i = 0; i < NC_Int - 1; i++)
        {
            derivative_data[i] = derivative_data[i + 1];
        }
#endif
        current_der = NC_Int - 1;
    }
    else
    {
        current_der = sample;
    }

    if (initialized || sample >= BUFFER_SIZE)
    {
#ifdef FP16_VECTORIATION 
        for (int i = 0; i < (BUFFER_SIZE) / 2; i += 1)
        {
            AV = *((INP_VTYPE *)&integrated_data[ 2 * i + 1]);
            integrated_data_v[i] = (INP_VTYPE)AV;//integrated_data_v[i + 1];
        }

#elif defined(FP8_VECTORIATION)
        for (int i = 0; i < (BUFFER_SIZE) / 4; i += 1)
        {
            AV = *((INP_VTYPE *)&integrated_data[ 4 * i + 1]);
            integrated_data_v[i] = (INP_VTYPE)AV;//integrated_data_v[i + 1];
        }
#else
        for (int i = 0; i < BUFFER_SIZE - 1; i++)
        {
            integrated_data[i] = integrated_data[i + 1];
        }
#endif
        current_int = BUFFER_SIZE - 1;
    }
    else
    {
        current_int = sample;
    }

    // Check validity of input data
    /*
    if (isfinite((dataType)input_data[1]) == 0)
    {
        input_data[1] = input_data[0];
    }
    */

    // Cancellation of DC component and normalization
    if (current_signal >= 1)
        signal_data[current_signal] = (dataType)input_data[1] - (dataType)input_data[0] + (dataType)(0.995) * (dataType)signal_data[current_signal - 1];

    else
        signal_data[current_signal] = 0;

    signal_data[current_signal] /= MAX_VALUE;

    // Terminate if we cannot apply the first filter
    if (current_lo < NC_Lo - 1)
        return 0.0f;

    // Low Pass Filtering
    //ATTENTION: IN FP8, WE USE FP16 FOR THE SIGNAL DATA; if you are using pure FP8, change the function here to single_convolution
    #ifdef FP8 
    lowpass_data[current_lo] = single_convolution(signal_data, current_signal + 1,h_Lo, NC_Lo);
    #else
        lowpass_data[current_lo] = single_convolution(signal_data, current_signal + 1, h_Lo, NC_Lo);
    #endif
    // High Pass Filtering
    highpass_data[current_hi] = single_convolution(lowpass_data, current_lo + 1, h_Hi, NC_Hi);
    // Derivative & squaring
    derivative_data[current_der] = single_convolution(highpass_data, current_hi + 1, h_Der, NC_Der);
    derivative_data[current_der] = derivative_data[current_der] * derivative_data[current_der];
    // Moving window integration
    integrated_data[current_int] = single_convolution(derivative_data, current_der + 1, h_Int, NC_Int);

    // Find R points
    dist_previous_peak = sample - previous_peak;
    dist_searchback_end = sample - searchback_end;
    if (dist_previous_peak < 0)
    {
        dist_previous_peak += MAX_SAMPLES;
    }
    if (dist_searchback_end < 0)
    {
        dist_searchback_end += MAX_SAMPLES;
    }


    // Check if a searchback is required
    if (dist_previous_peak > max_rr_width && dist_searchback_end > max_rr_width) // we use threshold2
    {

        searchback_end = sample;
        for (int i = 0; i <= current_int; i++)
        {
            // update peaki with current integrated value
            peaki = (qrsdataType)integrated_data[i];
            // Look for a QRS
            is_qrs = false;
            // Found a peak in searchback (threshold2)
            if (peaki > threshold2)
            {
                spki = (qrsdataType)(.750) * spki + (qrsdataType)(0.250) * peaki;
                is_qrs = true;
            }
            // Update data structures
            if (is_qrs)
            {
                if (peak_counter == 0 || dist_previous_peak >= min_rr_width)
                {
                    // Add a new peak
                    peak_counter = peak_counter + 1;
                    R_loc[(peak_counter - 1) % N_AVG] = sample;
                    // previous_peak = sample;
                }
                else if ((qrsdataType)integrated_data[0] < peaki)
                {
                    // Replace previous peak
                    R_loc[(peak_counter - 1) % N_AVG] = sample;
                    // previous_peak = sample;
                }
                previous_peak = sample;
            }
            else
            {
                // Update npki
                npki = (qrsdataType)(0.875) * npki + (qrsdataType)(0.125) * peaki;
            }
            // Adjust thresholds
            threshold1 = npki + (qrsdataType)(0.25) * (spki - npki);
            threshold2 = (qrsdataType)(0.5) * threshold1;
        }
    }
    else // we use threshold1
    {
        // update peaki with current integrated value
        peaki = (qrsdataType)integrated_data[current_int];
        // Look for a QRS
        is_qrs = false;
        // Found a peak (threshold1)
        if (peaki > threshold1)
        {
            spki = (qrsdataType)(0.875) * spki + (qrsdataType)(0.125) * peaki;
            is_qrs = true;
        }
        // Update data structures
        if (is_qrs)
        {
            int prev_peak_idx = current_int - dist_previous_peak;
            if (prev_peak_idx < 0)
                prev_peak_idx = 0;
            if (peak_counter == 0 || dist_previous_peak >= min_rr_width)
            {
                // Add a new peak
                peak_counter = peak_counter + 1;
                R_loc[(peak_counter - 1) % N_AVG] = sample;
                // previous_peak = sample;
            }
            else if ((qrsdataType)integrated_data[prev_peak_idx] < peaki)
            {
                // Replace previous peak
                R_loc[(peak_counter - 1) % N_AVG] = sample;
                // previous_peak = sample;
            }
            previous_peak = sample;
        }
        else
        {
            // Update npki
            npki = (qrsdataType)(0.875) * npki + (qrsdataType)(0.125) * peaki;
        }
        // Adjust thresholds
        threshold1 = npki + (qrsdataType)(0.25) * (spki - npki);
        threshold2 = (qrsdataType)(0.5) * threshold1;
    }

    // Compute the heart rate
    if (peak_counter >= N_AVG)
        peak_processing = 1;

    if (peak_processing)
    {
        hr = 0.0f;
        for (int i = 1; i < N_AVG; i++)
        {

            int idx = (((peak_counter - 1) % N_AVG) + 1 + i) % N_AVG;
            int temp = R_loc[idx] - R_loc[(idx + N_AVG - 1) % N_AVG];
            if (temp < 0)
            {
                temp += MAX_SAMPLES;
            }
            hr += temp;
        }
        hr /= (N_AVG - 1);
        hr = (hrdataType)(60) / (hrdataType)(hr / FS);
    }

    return hr;
}

void main_fn()
{
    hrdataType hr = 0;
    int sample = 0;

    #ifndef FABRIC
    pi_cl_team_barrier();
    #endif
    uint32_t core_id = pi_core_id(), cluster_id = pi_cluster_id();

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
    #endif // gap9
    #endif // stats
    // Start Power mesurement
    #if defined(__GAP9__)
    pi_pad_function_set(GPIOs, 1);
    pi_gpio_pin_configure(GPIOs, PI_GPIO_OUTPUT);
    pi_gpio_pin_write(GPIOs, 0);
    WRITE_GPIO(0);

    WRITE_GPIO(1);
    #endif

    // Global status
    peaki = 0, spki = 0, npki = 0, threshold1 = 0, threshold2 = 0;
    previous_peak = 0;
    peak_counter = 0;
    searchback_end = 0;
    initialized = 0;
    peak_processing = 0;

    // Processing
    while (sample < N)
    {
        hr = ecg_processing(input_data + sample, sample);
        #ifdef DEBUG
                printf("HR @ %d = %f and golden is %0.10f\n", sample, hr, reference[sample]);
        #endif
        if (sample < MAX_SAMPLES)
        {
            sample++;
        }
        else
        {
            sample = 0;
            initialized = 1;
        }
    }

    #ifndef FABRIC
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
}

#ifndef FABRIC
static void cluster_entry(void *arg)
{
    pi_cl_team_fork(NUM_CORES, main_fn, (void *)0x0);
}
#endif

int main()
{
    #if defined(__GAP9__)
    pi_time_wait_us(10000);

    pi_freq_set(PI_FREQ_DOMAIN_FC, 240*1000*1000);

    pi_time_wait_us(10000);

    pi_freq_set(PI_FREQ_DOMAIN_CL, 240*1000*1000);

    pi_time_wait_us(10000);
    #endif
    #ifdef FABRIC
        // printf("Hello from FC!\n");
        main_fn();
    #else
    struct pi_device cluster_dev = {0};
    struct pi_cluster_conf conf;
    struct pi_cluster_task cluster_task = {0};

    // task parameters allocation
    pi_cluster_task(&cluster_task, cluster_entry, NULL);
    // [OPTIONAL] specify the stack size for the task
    #if !defined(__GAP9__)
        cluster_task.stack_size = STACK_SIZE;

    #endif
    cluster_task.slave_stack_size = STACK_SIZE;

    // First open the cluster
    pi_cluster_conf_init(&conf);
    // conf.id=0;
    pi_open_from_conf(&cluster_dev, &conf);
    if (pi_cluster_open(&cluster_dev))
        return -1;

    // Then offload an entry point, this will get executed on the cluster controller
    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

    // closing of the cluster
    pi_cluster_close(&cluster_dev);
    #endif

    printf("Execution Done!\n");
    // Print the performance values if GAP9 is used
    #ifdef STATS
    #if defined(__GAP9__)
    #ifndef FABRIC
    for (uint32_t i = 0; i < ARCHI_CLUSTER_NB_PE; i++)
    {
        printf("[%d] Perf : %d cycles\n", i, perf_values[i]);
    }
    #else
        printf("Perf : %d cycles\n", perf_value);
    #endif
    #endif
    #endif

    return 0;
}
