#pragma once // Prevents multiple inclusions

#include <cuda_runtime.h>

__global__ void kernel0_embedding_layer(
    const long* input_ids,        
    const float* vector_token,   
    const float* position_token,     
    float* output_token,      
    int input_lenght,                  
    int n_embd                   
);

__global__ void kernel1_normalization_layer(
    float* input_embd,
    const float* gamma, 
    const float* beta,
    int n_embd,
    float epsilon
);

__global__ void kernel2_softmax_layer(
    float* input_scores, 
    const int input_size
);

__global__ void kernel3_add_vector_layer(
    const float* a, 
    const float* b, 
    float* c, 
    const size_t N
);

__global__ void kernel4_gelu_activation_layer(
    float* a, 
    const size_t N
);

__global__ void kernel_transpose_split_heads(
    const float* input,
    float* output,
    int seq_len,
    int n_heads,
    int head_dim
);

__global__ void kernel_transpose_merge_heads(
    const float* input,
    float* output,
    int seq_len,
    int n_heads,
    int head_dim
);