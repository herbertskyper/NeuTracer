#ifndef CUDA_OPERATIONS_H
#define CUDA_OPERATIONS_H

#ifdef __cplusplus
extern "C" {
#endif

// 向量相加函数声明
void vectorAdd(const float *a, const float *b, float *c, int n);

#ifdef __cplusplus
}
#endif

#endif
