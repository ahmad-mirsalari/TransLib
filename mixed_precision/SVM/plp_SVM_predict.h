#ifndef _SVM_PREDICT_H
#define _SVM_PREDICT_H

#include "defines.h"
//#define DATA_TYPE float

#ifdef __cplusplus
extern "C" {
#endif

 
/*typedef struct svm_node
{
	int index;
	DATA_TYPE value;
} svm_node;*/
 
enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */

 
 volatile typedef struct svm_model
{
	int SVM_TYPE;
	int KERNEL_TYPE;
	int DEGREE_SVM;
	FIL_TYPE GAMMA1;
	int COEF0_SVM;
	int SVS;
	int RHO_SIZE;
	int COEF_DIM;
	int F_DIM;
	int N_CLASS;
	int N_DEC_VALUES; 
	int *num_SVs;
	int *LABEL;
} svm_model;


void plp_SVM_linear(svm_model model, INP_TYPE *data_model, int *Pred, FIL_TYPE *sv_coef, FIL_TYPE *bias);
void plp_SVM_RBF(svm_model model, INP_TYPE *data_model, FIL_TYPE *sv_coef, FIL_TYPE *bias, INP_TYPE *x_ref, int *Pred);

 
#endif /* _LIBSVM_H */
