#include "pmsis.h"
#include "config.h"

#ifdef VECTORIAL
// Vectorized

#ifdef TRANSPOSE
// Transposed vectorized

#ifdef FP8
// Transposed vectorized FP8

void __attribute__((noinline)) matMul(MA_TYPE *__restrict__ A, MB_TYPE *__restrict__ B, OUT_TYPE *__restrict__ C, int M, int N, int P)
{

  #ifdef MIXED_VECTOR
  float temp[4];
  #else
  OUT_VTYPE temp, temp1, temp2, temp3;
  #endif

  MA_VTYPE Av;
  MB_VTYPE Bv0;
  MB_VTYPE Bv1;
  MB_VTYPE Bv2;
  MB_VTYPE Bv3;

  #ifndef MIXED_VECTOR
  OUT_VTYPE *Cv;
  #endif

  #ifndef FABRIC
  int core_id = pi_core_id();
  #else
  int core_id = 0;
  #endif

  int blockSize = (M + NUM_CORES - 1) / NUM_CORES;
  int start = core_id * blockSize;
  int end = start + blockSize < M ? start + blockSize : M;
  int vect_limit = N & ~0x3;
  int remainder_inloop = N & 0x3;
  int remainder_outloop = P & 0x3;

  for (int i = start; i < end; i++)
  {
    for (int j = 0; j < P; j += 4)
    {

      #ifdef MIXED_VECTOR
      memset(temp, 0, sizeof(temp));
      #else
      temp = (OUT_VTYPE){0, 0, 0, 0};
      temp1 = (OUT_VTYPE){0, 0, 0, 0};
      temp2 = (OUT_VTYPE){0, 0, 0, 0};
      temp3 = (OUT_VTYPE){0, 0, 0, 0};
      #endif

      // Manual unrolling
      for (int k = 0; k < vect_limit ; k += 4)
      {
        Av = *((MA_VTYPE *)&A[i * N + k]);
        Bv0 = *((MB_VTYPE *)&B[j * N + k]);
        Bv1 = *((MB_VTYPE *)&B[j * N + k + 1 * N]);
        Bv2 = *((MB_VTYPE *)&B[j * N + k + 2 * N]);
        Bv3 = *((MB_VTYPE *)&B[j * N + k + 3 * N]);

        #ifdef MIXED_VECTOR
        __asm__ __volatile__("vfdotpex.s.b %0, %1, %2" : "+f"(temp[0]) : "f"(Av), "f"(Bv0) :);
        __asm__ __volatile__("vfdotpex.s.b %0, %1, %2" : "+f"(temp[1]) : "f"(Av), "f"(Bv1) :);
        __asm__ __volatile__("vfdotpex.s.b %0, %1, %2" : "+f"(temp[2]) : "f"(Av), "f"(Bv2) :);
        __asm__ __volatile__("vfdotpex.s.b %0, %1, %2" : "+f"(temp[3]) : "f"(Av), "f"(Bv3) :);
        #else
        temp += (OUT_VTYPE) (Av * Bv0);
        temp1 += (OUT_VTYPE) (Av * Bv1);
        temp2 += (OUT_VTYPE) (Av * Bv2);
        temp3 += (OUT_VTYPE) (Av * Bv3);
        #endif
      }

      for (int k = N - remainder_inloop; k < N; k++ )
      { 

        #ifdef MIXED_VECTOR

        temp[0] += (float)A[ i * N + k] * (float)B[(j) * N + k];
        temp[1] += (float)A[ i * N + k] * (float)B[(j + 1) * N + k];
        temp[2] += (float)A[ i * N + k] * (float)B[ (j + 2) * N + k];
        temp[3] += (float)A[ i * N + k] * (float)B[(j + 3) * N + k];

        #else
        temp[0] += (OUT_TYPE)(A[ i * N + k]) * (OUT_TYPE)(B[(j) * N + k]);
        temp1[0] += (OUT_TYPE)(A[ i * N + k]) * (OUT_TYPE)(B[(j + 1) * N + k]);
        temp2[0] += (OUT_TYPE)(A[ i * N + k]) * (OUT_TYPE)(B[ (j + 2) * N + k]);
        temp3[0] += (OUT_TYPE)(A[ i * N + k]) * (OUT_TYPE)(B[(j + 3) * N + k]);
        #endif
      }
      #ifdef MIXED_VECTOR
          
      C[i * P + j] = (OUT_TYPE) temp[0];
      C[i * P + j + 1] = (OUT_TYPE) temp[1];
      C[i * P + j + 2] = (OUT_TYPE) temp[2];
      C[i * P + j + 3] = (OUT_TYPE) temp[3];

      #else
      Cv = (OUT_VTYPE *)&C[i * P + j];
      *Cv =(OUT_VTYPE ) {temp[0] + temp[1] + temp[2] + temp[3], temp1[0] + temp1[1] + temp1[2] + temp1[3], temp2[0] + temp2[1] + temp2[2] + temp2[3], temp3[0] + temp3[1] + temp3[2] + temp3[3]};
      #endif
    }
  }
  /// Leftover in P
  for (int j = P - remainder_outloop; j < P; j++)
  {
    for (int i = start; i < end; i++)
    {

      #ifdef MIXED_VECTOR
      float temp1 = 0.0f;
      #else
      temp = (OUT_VTYPE){0, 0, 0, 0};
      #endif

      // Manual unrolling
      for (int k = 0; k < vect_limit; k += 4)
      {

      Av = *((MA_VTYPE *)&A[i * N + k]);
      Bv0 = *((MB_VTYPE *)&B[j * N + k]);
      #ifdef MIXED_VECTOR
      __asm__ __volatile__("vfdotpex.s.b %0, %1, %2" : "+f"(temp1) : "f"(Av), "f"(Bv0) :);
      #else
      temp += (OUT_VTYPE) (Av * Bv0);
      #endif
      }

      for (int k = N - remainder_inloop; k < N; k++ )
      {
        #ifdef MIXED_VECTOR
        temp1 += (float)A[i * N + k] * (float)B[j * N + k];
        #else
        temp[0] += (OUT_TYPE)(A[i * N + k]) * (OUT_TYPE)(B[j * N + k]);
        #endif
      }
      #ifdef MIXED_VECTOR
      C[i * P + j] = (OUT_TYPE) temp1;
      #else
      C[i * P + j] = (OUT_TYPE)(temp[0] + temp[1] + temp[2] + temp[3]);
      #endif
    }
  }

#if NUM_CORES > 1
  pi_cl_team_barrier();
#endif
}

#else
// Transposed vectorized FP16 and FP16ALT
void __attribute__((noinline)) matMul(MA_TYPE *__restrict__ A, MB_TYPE *__restrict__ B, OUT_TYPE *__restrict__ C, int M, int N, int P)
{

  #ifdef MIXED_VECTOR
  float temp[2];
  #else
  OUT_VTYPE temp, temp1;
  #endif

  MA_VTYPE Av;
  MB_VTYPE Bv0;
  MB_VTYPE Bv1;
  
  #ifndef MIXED_VECTOR
  OUT_VTYPE *Cv, *Cv1;
  #endif

  #ifndef FABRIC
  int core_id = pi_core_id();
  #else
  int core_id = 0;
  #endif
  int blockSize = (M + NUM_CORES - 1) / NUM_CORES;
  int start = core_id * blockSize;
  int end = start + blockSize < M ? start + blockSize : M;

  for (int i = start; i < end; i++)
  {
    for (int j = 0; j < (P & 0xfffffffe); j += 2)
    {

      #ifdef MIXED_VECTOR
      temp[0] = temp[1] = 0;
      #else
      temp = (OUT_VTYPE){0, 0};
      temp1 = (OUT_VTYPE){0, 0};
      #endif
      // Manual unrolling
      for (int k = 0; k < (N & 0xfffffffe); k += 2)
      {
        Av = *((MA_VTYPE *)&A[i * N + k]);
        Bv0 = *((MB_VTYPE *)&B[j * N + k]);
        Bv1 = *((MB_VTYPE *)&B[j * N + N + k]);

        #ifdef MIXED_VECTOR
          #ifdef FP16
          __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(temp[0]) : "f"(Av), "f"(Bv0) :);
          __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(temp[1]) : "f"(Av), "f"(Bv1) :);
          #else
          __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(temp[0]) : "f"(Av), "f"(Bv0) :);
          __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(temp[1]) : "f"(Av), "f"(Bv1) :);
          #endif
        #else
        temp += (OUT_VTYPE) (Av * Bv0);
        temp1 += (OUT_VTYPE) (Av * Bv1);
        #endif
      }

      if (N & 0x00000001)
      {
        #ifdef MIXED_VECTOR
        temp[0] += (float)A[i * N + N - 1] * (float)B[(j + 1) * N - 1];
        temp[1] += (float)A[i * N + N - 1] * (float)B[(j + 2) * N - 1];
        #else
        temp[0] += (OUT_TYPE)(A[i * N + N - 1]) * (OUT_TYPE)(B[(j + 1) * N - 1]);
        temp1[0] += (OUT_TYPE)(A[i * N + N - 1]) * (OUT_TYPE)(B[(j + 2) * N - 1]);
        #endif
      }

      #ifdef MIXED_VECTOR
      C[i * P + j] = (OUT_TYPE) temp[0];
      C[i * P + j + 1] = (OUT_TYPE) temp[1];
      #else
      Cv = (OUT_VTYPE *)&C[i * P + j];
      *Cv = (OUT_VTYPE) {temp[0]+temp[1], temp1[0]+temp1[1]};
      #endif
    }
  }
  /// Leftover in P
  if (P & 0x00000001)
  {
    for (int i = start; i < end; i++)
    {

      #ifdef MIXED_VECTOR
      float temp2 = 0;
      #else
      temp = (OUT_VTYPE){0, 0};
      #endif

      // Manual unrolling
      for (int k = 0; k < (N & 0xfffffffe); k += 2)
      {
        Av = *((MA_VTYPE *)&A[i * N + k]);
        Bv0 = *((MB_VTYPE *)&B[(P - 1) * N + k]);
        #ifdef MIXED_VECTOR
        #ifdef FP16
        __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(temp2) : "f"(Av), "f"(Bv0) :);
        #else
        __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(temp2) : "f"(Av), "f"(Bv0) :);
        #endif
        #else
        temp += (OUT_VTYPE) (Av * Bv0);
        #endif
      }
      if (N & 0x00000001)
      {
        #ifdef MIXED_VECTOR
        temp2 += (float)A[i * N + N - 1] * (float)B[(P - 1) * N + N - 1];
        #else
        temp[0] += (OUT_TYPE)(A[i * N + N - 1]) * (OUT_TYPE)(B[(P - 1) * N + N - 1]);
        #endif
      }

      #ifdef MIXED_VECTOR
      C[i * P + (P - 1)] = (OUT_TYPE)temp2;
      #else
      C[i * P + (P - 1)] = (OUT_TYPE) {temp[0]+temp[1]};
      #endif
    }
  }

#if NUM_CORES > 1
  pi_cl_team_barrier();
#endif
}
#endif

#else
// Non-transposed vectorized FP8, FP16, and FP16ALT
#ifdef FP8
// Non-transposed vectorized FP8
void __attribute__((noinline)) matMul(MA_TYPE *__restrict__ A, MB_TYPE *__restrict__ B, OUT_TYPE *__restrict__ C, int M, int N, int P)
{

  #ifdef MIXED_VECTOR
  float temp[4];
  #else
  OUT_VTYPE temp;
  #endif
  
  MA_VTYPE Av;
  MB_VTYPE Bv0;
  MB_VTYPE Bv1;
  MB_VTYPE Bv2;
  MB_VTYPE Bv3;
  
  #ifndef MIXED_VECTOR
  OUT_VTYPE *Cv;
  #endif

  #ifndef FABRIC
  int core_id = pi_core_id();
  #else
  int core_id = 0;
  #endif
  int blockSize = (M + NUM_CORES - 1) / NUM_CORES;
  int start = core_id * blockSize;
  int end = start + blockSize < M ? start + blockSize : M;
  int vect_limit = N & ~0x3;
  int remainder_inloop = N & 0x3;
  int remainder_outloop = P & 0x3;

  for (int i = start; i < end; i++)
  {
    for (int j = 0; j < P; j += 4)
    {

      #ifdef MIXED_VECTOR
      memset(temp, 0, sizeof temp);
      #else
      temp = (OUT_VTYPE){0, 0, 0, 0};
      #endif

      // Manual unrolling
      for (int k = 0; k < vect_limit ; k += 4)
      {

        Av = *((MA_VTYPE *)&A[i * N + k]);
        Bv0 = *((MB_VTYPE *)&B[k * P + j]);
        Bv1 = *((MB_VTYPE *)&B[k * P + j + 1 * P]);
        Bv2 = *((MB_VTYPE *)&B[k * P + j + 2 * P]);
        Bv3 = *((MB_VTYPE *)&B[k * P + j + 3 * P]);

        #ifdef MIXED_VECTOR
        MB_VTYPE t0 = __builtin_shuffle(Bv0, Bv1, (v4s){0,4,1,5});
        MB_VTYPE t1 = __builtin_shuffle(Bv2, Bv3, (v4s){0,4,1,5});
        MB_VTYPE t2 = __builtin_shuffle(Bv0, Bv1, (v4s){2,6,3,7});
        MB_VTYPE t3 = __builtin_shuffle(Bv2, Bv3, (v4s){2,6,3,7});

        MB_VTYPE Val1 = __builtin_shuffle(t0, t1, (v4s){0,1,4,5});
        MB_VTYPE Val2 = __builtin_shuffle(t0, t1, (v4s){2,3,6,7});
        MB_VTYPE Val3 = __builtin_shuffle(t2, t3, (v4s){0,1,4,5});
        MB_VTYPE Val4 = __builtin_shuffle(t2, t3, (v4s){2,3,6,7});

        __asm__ __volatile__("vfdotpex.s.b %0, %1, %2" : "+f"(temp[0]) : "f"(Av), "f"(Val1) :);
        __asm__ __volatile__("vfdotpex.s.b %0, %1, %2" : "+f"(temp[1]) : "f"(Av), "f"(Val2) :);
        __asm__ __volatile__("vfdotpex.s.b %0, %1, %2" : "+f"(temp[2]) : "f"(Av), "f"(Val3) :);
        __asm__ __volatile__("vfdotpex.s.b %0, %1, %2" : "+f"(temp[3]) : "f"(Av), "f"(Val4) :);
        #else
        temp += (OUT_VTYPE)(__builtin_shuffle(Av, (v4s){0, 0, 0, 0})) * Bv0;
        temp += (OUT_VTYPE)(__builtin_shuffle(Av, (v4s){1, 1, 1, 1})) * Bv1;
        temp += (OUT_VTYPE)(__builtin_shuffle(Av, (v4s){2, 2, 2, 2})) * Bv2;
        temp += (OUT_VTYPE)(__builtin_shuffle(Av, (v4s){3, 3, 3, 3})) * Bv3;
        #endif
      }

      for (int k = N - remainder_inloop ; k < N; k++ )
      {

        Av = *((MA_VTYPE *)&A[i * N + k]);
        Bv0 = *((MB_VTYPE *)&B[k * P + j]);
        MA_VTYPE t0 = __builtin_shuffle(Av, (v4s){0, 0, 0, 0});

        #ifdef MIXED_VECTOR
        temp[0] += (float)A[i * N + k] * (float)B[k * P + j];
        temp[1] += (float)A[i * N + k] * (float)B[k * P + j + 1];
        temp[2] += (float)A[i * N + k] * (float)B[k * P + j + 2];
        temp[3] += (float)A[i * N + k] * (float)B[k * P + j + 3];
        #else
        temp += (OUT_VTYPE)(t0 * Bv0);
        // temp[0] += A[i * N + k] * B[ k * P + j];
        // temp[1] += A[i * N + k] * B[ k * P + j + 1];
        // temp[2] += A[i * N + k] * B[ k * P + j + 2];
        // temp[3] += A[i * N + k] * B[ k * P + j + 3];
        #endif
      }
      #ifdef MIXED_VECTOR
      C[i * P + j] = (OUT_TYPE) temp[0];
      C[i * P + j + 1] = (OUT_TYPE) temp[1];
      C[i * P + j + 2] = (OUT_TYPE) temp[2];
      C[i * P + j + 3] = (OUT_TYPE) temp[3];
      #else
      Cv = (OUT_VTYPE *)&C[i * P + j];
      *Cv = temp;
      #endif
    }
  }
  /// Leftover in P
  for (int j = P - remainder_outloop; j < P; j++)
  {
    for (int i = start; i < end; i++)
    {
      #ifdef MIXED_VECTOR
      float temp1 = 0.0f;
      #else
      OUT_TYPE temp1 = 0;
      #endif
      // Manual unrolling
      for (int k = 0; k < (N & 0xfffffffe); k += 2)
      {
        #ifdef MIXED_VECTOR
        temp1 += (float)A[i * N + k] * (float)B[k * P + j];
        temp1 += (float)A[i * N + k + 1] * (float)B[k * P + j + P];
        #else
        temp1 += (OUT_TYPE)(A[i * N + k]) * (OUT_TYPE)(B[k * P + j]);
        temp1 += (OUT_TYPE)(A[i * N + k + 1]) * (OUT_TYPE)(B[k * P + j + P]);
        #endif
      }
      if (N & 0x00000001)
      {

        #ifdef MIXED_VECTOR
        temp1 += (float)A[i * N + N - 1] * (float)B[(N - 1) * P + j];
        #else
        temp1 += (OUT_TYPE)(A[i * N + N - 1]) * (OUT_TYPE)(B[(N - 1) * P + j]);
        #endif
      }
      C[i * P + j] = (OUT_TYPE)(temp1);
    }
  }

#if NUM_CORES > 1
  pi_cl_team_barrier();
#endif
}

#else
// Non-transposed vectorized FP16 and FP16ALT
void __attribute__((noinline)) matMul(MA_TYPE *__restrict__ A, MB_TYPE *__restrict__ B, OUT_TYPE *__restrict__ C, int M, int N, int P)
{

  #ifdef MIXED_VECTOR
  float temp[2];
  #else
  OUT_VTYPE temp, temp1;
  #endif
  MA_VTYPE Av;
  MB_VTYPE Bv0;
  MB_VTYPE Bv1;
  #ifndef MIXED_VECTOR
  OUT_VTYPE *Cv, *Cv1;
  #endif

  #ifndef FABRIC
  int core_id = pi_core_id();
  #else
  int core_id = 0;
  #endif
  int blockSize = (M + NUM_CORES - 1) / NUM_CORES;
  int start = core_id * blockSize;
  int end = start + blockSize < M ? start + blockSize : M;

  for (int i = start; i < end; i++)
  {
    for (int j = 0; j < (P & 0xfffffffe); j += 2)
    {
      #ifdef MIXED_VECTOR
      temp[0] = temp[1] = 0;
      #else
      temp = (OUT_VTYPE){0, 0};
      temp1 = (OUT_VTYPE){0, 0};
      #endif

      // Manual unrolling
      for (int k = 0; k < (N & 0xfffffffe); k += 2)
      {
        Av = *((MA_VTYPE *)&A[i * N + k]);
        Bv0 = *((MB_VTYPE *)&B[k * P + j]);
        Bv1 = *((MB_VTYPE *)&B[k * P + P + j]);

        #ifdef MIXED_VECTOR
        MB_VTYPE Val1 = __builtin_shuffle(Bv0, Bv1, (v2s){0, 2});
        MB_VTYPE Val2 = __builtin_shuffle(Bv0, Bv1, (v2s){1, 3});
          #ifdef FP16
          __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(temp[0]) : "f"(Av), "f"(Val1) :);
          __asm__ __volatile__("vfdotpex.s.h %0, %1, %2" : "+f"(temp[1]) : "f"(Av), "f"(Val2) :);
          #else
          __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(temp[0]) : "f"(Av), "f"(Val1) :);
          __asm__ __volatile__("vfdotpex.s.ah %0, %1, %2" : "+f"(temp[1]) : "f"(Av), "f"(Val2) :);
          #endif
        #else
        temp += (OUT_VTYPE)(__builtin_shuffle(Av, (v2s){0, 0})) * Bv0;
        temp += (OUT_VTYPE)(__builtin_shuffle(Av, (v2s){1, 1})) * Bv1;
        #endif
      }


      if (N & 0x00000001)
      {
        Av = *((MA_VTYPE *)&A[i * N + N - 1]);
        Bv0 = *((MB_VTYPE *)&B[(N - 1) * P + j]);
        #ifdef MIXED_VECTOR
        temp[0] += (float)A[i * N + N - 1] * (float)B[(N - 1) * P + j];
        temp[1] += (float)A[i * N + N - 1] * (float)B[(N - 1) * P + j + 1];
        #else
        MA_VTYPE Val1 = __builtin_shuffle(Av, (v2s){0, 0});
        temp += (OUT_VTYPE)(Val1 * Bv0);
        // temp[0] += A[i * N + N - 1] * B[(N - 1) * P + j];
        // temp[1] += A[i * N + N - 1] * B[(N - 1) * P + j + 1];
        #endif
      }
      #ifdef MIXED_VECTOR
      C[i * P + j] = (OUT_TYPE) temp[0];
      C[i * P + j + 1] = (OUT_TYPE) temp[1];
      #else
      Cv = (OUT_VTYPE *)&C[i * P + j];
      *Cv = temp;
      #endif
    }
  }
  /// Leftover in P
  if (P & 0x00000001)
  {
    for (int i = start; i < end; i++)
    {

      #ifdef MIXED_VECTOR
      float temp1 = 0.0f;
      #else
      OUT_TYPE temp1 = 0;
      #endif
      // Manual unrolling
      for (int k = 0; k < (N & 0xfffffffe); k += 2)
      {
        #ifdef MIXED_VECTOR
        temp1 += (float)A[i * N + k] * (float)B[k * P + P - 1];
        temp1 += (float)A[i * N + k + 1] * (float)B[k * P + P - 1 + P];
        #else
        temp1 += (OUT_TYPE)A[i * N + k] * (OUT_TYPE)B[k * P + P - 1];
        temp1 += (OUT_TYPE)A[i * N + k + 1] * (OUT_TYPE)B[k * P + P - 1 + P];
        #endif
      }
      if (N & 0x00000001)
      {
        #ifdef MIXED_VECTOR
        temp1 += (float)A[i * N + N - 1] * (float)B[(N - 1) * P + P - 1];
        #else
        temp1 += (OUT_TYPE)(A[i * N + N - 1]) * (OUT_TYPE)(B[(N - 1) * P + P - 1]);
        #endif
      }

      C[i * P + (P - 1)] = (OUT_TYPE)temp1;
    }
  }

#if NUM_CORES > 1
  pi_cl_team_barrier();
#endif
}
#endif

#endif
#else
// Non-vectorized version
#ifdef TRANSPOSE
void __attribute__((noinline)) matMul(MA_TYPE *__restrict__ A, MB_TYPE *__restrict__ B, OUT_TYPE *__restrict__ C, int M, int N, int P)
{
  #ifndef FABRIC
  int core_id = pi_core_id();
  #else
  int core_id = 0;
  #endif

  #ifdef HWMIXED 
  float temp = 0.0f;
  #else
  OUT_TYPE temp = 0;
  #endif
  int blockSize = (M + NUM_CORES - 1) / NUM_CORES;
  int start = core_id * blockSize;
  int end = start + blockSize < M ? start + blockSize : M;

  for (int i = start; i < end; i++)
  {
    for (int j = 0; j < P; j++)
    {

      temp = 0;

      // Manual unrolling
      for (int k = 0; k < (N & 0xfffffffe); k += 2)
      {

        #ifdef HWMIXED
        temp += (float)A[i * N + k] * (float)B[j * N + k];
        temp += (float)A[i * N + k + 1] * (float)B[j * N + k + 1];
        #else
        temp += (OUT_TYPE)(A[i * N + k]) * (OUT_TYPE)(B[j * N + k]);
        temp += (OUT_TYPE)(A[i * N + k + 1]) * (OUT_TYPE)(B[j * N + k + 1]);
        #endif
      }

      // if (N & 0x00000001)
      // {
      //       temp += (OUT_TYPE)(A[i * N + N - 1] * B[j * N + (N - 1)]);
      // }
      C[i * P + j] = (OUT_TYPE)(temp);
    }
  }
  // Leftover on N

  if (N & 0x00000001)
  {
    for (int i = start; i < end; i++)
    {
      for (int j = 0; j < P; j++)
      {
        #ifdef HWMIXED
         temp = (float)A[i * N + N - 1] * (float) B[j * N + (N - 1)];
         C[i * P + j] += (OUT_TYPE) temp;
        #else
        C[i * P + j] += (OUT_TYPE)(A[i * N + N - 1]) * (OUT_TYPE)(B[j * N + (N - 1)]);
        #endif
      }
    }
  }

#if NUM_CORES > 1
  pi_cl_team_barrier();
#endif
}

#else
// Non-transposed non-vectorized 
void __attribute__((noinline)) matMul(MA_TYPE *__restrict__ A, MB_TYPE *__restrict__ B, OUT_TYPE *__restrict__ C, int M, int N, int P)
{
  
  #ifndef FABRIC
  int core_id = pi_core_id();
  #else
  int core_id = 0;
  #endif

  #ifdef HWMIXED 
  float temp = 0.0f;
  #else
  OUT_TYPE temp = 0;
  #endif

  int blockSize = (M + NUM_CORES - 1) / NUM_CORES;
  int start = core_id * blockSize;
  int end = start + blockSize < M ? start + blockSize : M;

  for (int i = start; i < end; i++)
  {
    for (int j = 0; j < P; j++)
    {
      temp = 0;

      // Manual unrolling
      for (int k = 0; k < (N & 0xfffffffe); k += 2)
      {
        #ifdef HWMIXED
        temp += (float)A[i * N + k] * (float)B[k * P + j];
        temp += (float)A[i * N + k + 1] * (float)B[(k + 1) * P + j];
        #else
        temp += (OUT_TYPE)(A[i * N + k]) * (OUT_TYPE)(B[k * P + j]);
        temp += (OUT_TYPE)(A[i * N + k + 1]) * (OUT_TYPE)(B[(k + 1) * P + j]);
        #endif

      }
      C[i * P + j] = (OUT_TYPE)(temp);
    }
  }
  // Leftover on N

  if (N & 0x00000001)
  {
    for (int i = start; i < end; i++)
    {
      for (int j = 0; j < P; j++)
      {
        #ifdef HWMIXED
        temp = (float) A[i * N + N - 1] * (float)B[(N - 1) * P + j];
        C[i * P + j] += (OUT_TYPE) temp;
        #else
        C[i * P + j] += (OUT_TYPE)(A[i * N + N - 1]) * (OUT_TYPE)B[(N - 1) * P + j];
        #endif
      }
    }
  }

#if NUM_CORES > 1
  pi_cl_team_barrier();
#endif
}
#endif
#endif