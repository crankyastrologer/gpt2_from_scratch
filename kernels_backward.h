#pragma once
#include <cuda_runtime.h>


__global__ void layer_norm_backward(
    const float* d_out,    // Incoming gradient (d_y)
    const float* input,    // Original input (x)
    const float* gamma,    // Scale parameter
    float* d_input,        // Result (dx)
    int n_embed,
    float epsilon

);
__global__ void fused_masked_softmax_backward(
    const float* grad_output,   // Incoming gradient (dY)
    const float* softmax_output,// Original Softmax output (S)
    float* grad_input,          // Result (dX)
    const int seq_len           // 1024
);
__global__ void gelu_backwards_kernel(
    float* d_in,
    const float* d_out, 
    const float* x,
    const size_t N
);
__global__ void kernel_transpose_reverse_merge_heads_layer(const float *dK, const float*dV,const float*dQ, float *out,int seq_len,int n_embed
    ,const int head_dim, const int n_head, const int batch_size);
__global__ void bias_layer(float * in, float *out,const int batch_size, const int seq_len, const int n_embed);
__global__ void my_adamw_kernel(
    float* params,          // The weights (p)
    const float* grads,     // The gradients (g)
    float* m_memory,        // First moment memory (m)
    float* v_memory,        // Second moment memory (v)
    const float lr,         // Learning rate
    const float beta1,      
    const float beta2,      
    const float eps,        
    const float weight_decay, 
    const float bias_correction_1, // Precalculated (1 - beta1^t) on CPU
    const float bias_correction_2, // Precalculated (1 - beta2^t) on CPU
    const size_t N                 // Total number of elements
);
__global__ void zero_grad_kernel(float * in,const int size);
__global__ void cross_entropy_backward_kernel(float* d_logits, const long* target,
const int vocab_size, const int total_tokens);