#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include "kernels.h"
#include "kernels_backward.h"
#define SEQ_LEN 1024
#define BLOCK_SIZE 256
#define N_EMBED 768
#define N_HEAD 12
#define N_LAYER 12
#define VOCAB_SIZE 50276
#define FFN_DIM 3072
#define HEAD_DIM 64

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

int main(){
    printf("starting gpt2 training loop");
    int batch_size = 1;
    int steps = 100;
    int seq_len = SEQ_LEN;
    int n_embed = N_EMBED;
    int head_dim = HEAD_DIM;
    size_t total_tokens = batch_size * seq_len;
    size_t total_elements_embed = total_tokens * n_embed;
    size_t total_elements_ffn = total_tokens * FFN_DIM;   
    float* d_final_ln_input;
    cudaMalloc(&d_final_ln_input, total_elements_embed*sizeof(float));
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    long* h_input_ids = (long*)malloc(total_tokens * sizeof(long));
    long* d_input_ids;
    cudaMalloc(&d_input_ids, total_tokens * sizeof(long));
    float* d_token_embed_matrix;
    float* d_pos_embed_matrix;
    cudaMalloc(&d_token_embed_matrix, VOCAB_SIZE * n_embed * sizeof(float));
    cudaMalloc(&d_pos_embed_matrix, seq_len * n_embed * sizeof(float));
    // Main data buffer
    float* d_model_output; // This buffer holds the state: [B*S, E]
    cudaMalloc(&d_model_output, total_elements_embed * sizeof(float));
    
    // Intermediate buffers
    float* d_residual_buffer[N_LAYER*2]; // For residual connections: [B*S, E]
    for(int i = 0;i<N_LAYER*2;i++)
        cudaMalloc(&d_residual_buffer[i], total_elements_embed * sizeof(float));
    
    // (Mock) LayerNorm Weights
    float* pre_gelu_buffer[N_LAYER];
    float* d_ffn_hidden_grad;
    cudaMalloc(&d_ffn_hidden_grad,total_elements_ffn*sizeof(float));
    float* d_ln_attn_gamma[N_LAYER];
    float* d_ln_ffn_gamma[N_LAYER];
    float* d_ln_final_gamma;
    float* d_ln_ffn_beta[N_LAYER],*d_ln_attn_beta[N_LAYER];
    float* d_ln_final_beta;
cudaMalloc(&d_ln_final_beta, n_embed * sizeof(float));
    cudaMalloc(&d_ln_final_gamma, n_embed * sizeof(float));
    
float *d_layer_Wq[N_LAYER];
        
        float  *d_layer_Wk[N_LAYER], *d_layer_Wv[N_LAYER], *d_layer_Wo[N_LAYER];
    
        float *d_Q[N_LAYER], *d_K[N_LAYER], *d_V[N_LAYER];
    
        float *d_Q_split[N_LAYER], *d_K_split[N_LAYER], *d_V_split[N_LAYER];
    
         float *d_scores[N_LAYER];
size_t scores_size = (size_t)batch_size * N_HEAD * seq_len * seq_len;
    float *d_attn_output;
    cudaMalloc(&d_attn_output, total_tokens * n_embed * sizeof(float));
        float *d_attn_output_split;
    cudaMalloc(&d_attn_output_split, total_tokens * n_embed * sizeof(float));
    float *ffn1[N_LAYER],*ffn2[N_LAYER],*ffh[N_LAYER];
    
    float* d_unembedding_matrix;
    cudaMalloc(&d_unembedding_matrix, n_embed * VOCAB_SIZE * sizeof(float));
    float alpha = 1.0f;
        float beta = 0.0f;
    float * d_unembedding_matrix_grid;
    cudaMalloc(&d_unembedding_matrix_grid, n_embed*VOCAB_SIZE*sizeof(float));
    float * d_model_output_grad;
    cudaMalloc(&d_model_output_grad, total_elements_embed*sizeof(float));
    printf("Memory allocated...\n");
    for(int i = 0;i<N_LAYER;i++)
    {
    cudaMalloc(&d_ln_attn_gamma[i], n_embed * sizeof(float));
    cudaMalloc(&d_ln_ffn_gamma[i], n_embed * sizeof(float));
    cudaMalloc(&d_ln_ffn_beta[i], n_embed * sizeof(float));
    cudaMalloc(&d_ln_attn_beta[i], n_embed * sizeof(float));
    cudaMalloc(&d_layer_Wk[i], n_embed * n_embed * sizeof(float));
    cudaMalloc(&d_layer_Wv[i], n_embed * n_embed * sizeof(float));
    cudaMalloc(&d_layer_Wo[i], n_embed * n_embed * sizeof(float));
    cudaMalloc(&d_layer_Wq[i],n_embed*n_embed*sizeof(float));
    cudaMalloc(&d_Q[i], total_tokens * n_embed * sizeof(float));
    cudaMalloc(&d_K[i], total_tokens * n_embed * sizeof(float));
    cudaMalloc(&d_V[i], total_tokens * n_embed * sizeof(float));
    cudaMalloc(&d_Q_split[i], total_tokens * n_embed * sizeof(float));
    cudaMalloc(&d_K_split[i], total_tokens * n_embed * sizeof(float));
    cudaMalloc(&d_V_split[i], total_tokens * n_embed * sizeof(float));
    cudaMalloc(&ffn1[i],n_embed*FFN_DIM*sizeof(float));
    cudaMalloc(&ffn2[i],FFN_DIM*n_embed*sizeof(float));
    cudaMalloc(&d_scores[i], scores_size * sizeof(float));
        cudaMalloc(&ffh[i],total_tokens*FFN_DIM*sizeof(float));
    cudaMalloc(&pre_gelu_buffer[i],n_embed*sizeof(float));

    }

    // (Mock) Output
    float* d_logits;
    cudaMalloc(&d_logits, total_tokens * VOCAB_SIZE * sizeof(float));
    
    printf("Memory allocated...\n");

    // --- 4. Initialize Mock Data (Example) ---
    for(int i = 0; i < total_tokens; i++) h_input_ids[i] = 1; // Just use token "1"
    cudaMemcpy(d_input_ids, h_input_ids, total_tokens * sizeof(long), cudaMemcpyHostToDevice);
    // (In real life, we'd cudaMemcpy all the pre-trained weights)

    // --- 5. EXECUTE THE FORWARD PASS ---
    for(int i = 0;i<steps;i++)
    {// --- LAUNCH KERNEL 1: FUSED EMBEDDING (Layer 0) ---
    printf("Running Layer 0: Embedding...\n");
    kernel0_embedding_layer<<<total_tokens, BLOCK_SIZE>>>(
        d_input_ids, d_token_embed_matrix, d_pos_embed_matrix, 
        d_model_output, seq_len, n_embed
    );cudaDeviceSynchronize();
    cudaCheckErrors("Layer 0 Embedding failed");

// --- DECODER LOOP ---
    for (int layer = 0; layer < N_LAYER; layer++) {
        printf("Running Decoder Layer %d...\n", layer);
        
        // Save for residual: d_residual_buffer = d_model_output
        cudaMemcpy(d_residual_buffer[layer*2], d_model_output, total_elements_embed * sizeof(float), cudaMemcpyDeviceToDevice);
        
        // --- LAUNCH KERNEL 2: LAYER NORM (Pre-Attention) ---
        kernel1_normalization_layer<<<total_tokens, BLOCK_SIZE>>>(
            d_model_output, d_ln_attn_gamma[layer], d_ln_attn_beta[layer], n_embed, 1e-5f
        );cudaDeviceSynchronize();
        
        cudaCheckErrors("Layer 1 (Pre-Attention) LayerNorm failed");

        // --- ATTENTION BLOCK ---
        // TODO: Launch cuBLAS GEMM calls for Q, K, V
        
        cublasSgemm(cublas_handle,CUBLAS_OP_N,CUBLAS_OP_N,
            n_embed,
            total_tokens,
            n_embed,
            &alpha,
            d_layer_Wq[layer],
            n_embed,
            d_model_output,
            n_embed,
            &beta,
            d_Q[layer],
            n_embed
        
        );cudaDeviceSynchronize();
        cublasSgemm(cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n_embed, total_tokens, n_embed,
            &alpha,
            d_layer_Wk[layer],    // Weights for K
            n_embed,
            d_model_output, // Same input
            n_embed,
            &beta,
            d_K[layer],           // K Result buffer
            n_embed
        );cudaDeviceSynchronize();

        // V
        cublasSgemm(cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n_embed, total_tokens, n_embed,
            &alpha,
            d_layer_Wv[layer],    // Weights for V
            n_embed,
            d_model_output, // Same input
            n_embed,
            &beta,
            d_V[layer],           // V Result buffer
            n_embed
        );cudaDeviceSynchronize();
        cublasSgemm(cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            total_tokens,  // n
            total_tokens,  // m
            n_embed,       // k
            &alpha,
            d_K[layer],           // B (K)
            n_embed,       // ldb
            d_Q[layer],           // A (Q)
            n_embed,       // lda
            &beta,
            d_scores[layer],      // C
            total_tokens   // ldc
        );cudaDeviceSynchronize();
        // TODO: Reshape for Multi-Head
        // TODO: Launch batched cuBLAS GEMM for Q*K^T (Scores)
        int grid_trans = total_tokens;
        kernel_transpose_split_heads_layer<<<grid_trans, BLOCK_SIZE>>>(d_Q[layer], d_Q_split[layer], seq_len, N_HEAD, head_dim);
        kernel_transpose_split_heads_layer<<<grid_trans, BLOCK_SIZE>>>(d_K[layer], d_K_split[layer], seq_len, N_HEAD, head_dim);
        kernel_transpose_split_heads_layer<<<grid_trans, BLOCK_SIZE>>>(d_V[layer], d_V_split[layer], seq_len, N_HEAD, head_dim);
        long long stride_Q=seq_len*head_dim;
        long long stride_K=seq_len*head_dim;
        long long stride_Scores = seq_len*seq_len;
        cublasSgemmStridedBatched(
            cublas_handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            seq_len,seq_len,
            head_dim,
            &alpha,
            d_K_split[layer],head_dim, stride_K,
            d_Q_split[layer], head_dim, stride_Q,
            &beta,
            d_scores[layer], seq_len, stride_Scores,
            N_HEAD


        );
        cudaCheckErrors("Batched Q*K^T failed");
        
        // --- LAUNCH KERNEL 3: FUSED MASKED SOFTMAX ---
        // (This assumes a 'd_scores' buffer was calculated and has shape [B*H*S, S])
        // int grid_size_softmax = total_tokens * N_HEAD;
        // fused_masked_softmax<<<grid_size_softmax, BLOCK_SIZE>>>(d_scores, seq_len);
        kernel2_softmax_layer<<<N_HEAD * seq_len, BLOCK_SIZE>>>(d_scores[layer],seq_len);cudaDeviceSynchronize();
        cudaCheckErrors("Softmax Failed");
        // TODO: Launch batched cuBLAS GEMM for Scores * V
        // TODO: Reshape back from Multi-Head
        // TODO: Launch cuBLAS GEMM for final output projection
        long long stride_V = seq_len*head_dim;
        long long stride_Out = seq_len*seq_len;
        cublasSgemmStridedBatched(cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        head_dim,
        seq_len,
        seq_len,
        &alpha,
        d_V_split[layer], head_dim, stride_V,
        d_scores[layer], seq_len, stride_Scores,
        &beta,
        d_attn_output_split, head_dim,stride_Out,N_HEAD
    );
    cudaCheckErrors("batched scores*V failed");
    kernel_transpose_merge_heads_layer<<<grid_trans, BLOCK_SIZE>>>(d_attn_output_split, d_attn_output, seq_len, N_HEAD, head_dim);
        cudaDeviceSynchronize();
        cudaCheckErrors("Transpose Merge failed");
         
        // 4. Final Projection (Output Weights)
        // Output = Attn_Output * W_o
        cublasSgemm(cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n_embed, total_tokens, n_embed,
            &alpha,
            d_layer_Wo[layer],    // Output Projection Weights
            n_embed,
            d_attn_output, // Input
            n_embed,
            &beta,
            d_model_output,// Write back to main buffer (or a temp one)
            n_embed
        );cudaDeviceSynchronize();
        // --- LAUNCH KERNEL 4: RESIDUAL ADD (Attention) ---
       int grid_size_embed = (total_elements_embed + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernel3_add_vector_layer<<<grid_size_embed, BLOCK_SIZE>>>(
            d_residual_buffer[layer*2], d_model_output, d_model_output, total_elements_embed
        );cudaDeviceSynchronize();
        cudaCheckErrors("Layer 4 (Residual 1) failed");

        // --- FFN BLOCK ---
        // Save for residual: d_residual_buffer = d_model_output
        cudaMemcpy(d_residual_buffer[layer*2+1], d_model_output, total_elements_embed * sizeof(float), cudaMemcpyDeviceToDevice);
        

        // --- LAUNCH KERNEL 2: LAYER NORM (Pre-FFN) ---
        kernel1_normalization_layer<<<total_tokens, BLOCK_SIZE>>>(
            d_model_output, d_ln_ffn_gamma[layer], d_ln_ffn_beta[layer], n_embed, 1e-5f
        );cudaDeviceSynchronize();
        cudaCheckErrors("Layer 1 (Pre-FFN) LayerNorm failed");
        
        // TODO: Launch cuBLAS GEMM for FFN Expand (d_model_output -> d_ffn_hidden)
        cublasSgemm(cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            FFN_DIM, total_tokens, n_embed,
            &alpha,
            ffn1[layer],
            FFN_DIM,
            d_model_output,
            n_embed,
            &beta,
            ffh[layer],
            FFN_DIM
        );cudaDeviceSynchronize();
        
        // --- LAUNCH KERNEL 5: GELU ---
        // (This assumes d_ffn_hidden is [B*S, FFN_DIM])
        // gelu_kernel<<<(total_elements_ffn + 255) / 256, BLOCK_SIZE>>>(d_ffn_hidden, total_elements_ffn);
        int grid_size_ffn = (total_elements_ffn+BLOCK_SIZE-1)/BLOCK_SIZE;
        kernel4_gelu_activation_layer<<<grid_size_ffn, BLOCK_SIZE>>>(ffh[layer],total_elements_ffn);
        cudaCheckErrors("GELU failed");
        // TODO: Launch cuBLAS GEMM for FFN Contract (d_ffn_hidden -> d_model_output)
        cublasSgemm(cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n_embed, total_tokens, FFN_DIM,
            &alpha,
            ffn2[layer], // d_layer_ffn2_weight
            n_embed,
            ffh[layer], // d_ffn_hidden
            FFN_DIM,
            &beta,
            d_model_output, // Write back to main buffer
            n_embed
        );cudaDeviceSynchronize();
        // --- LAUNCH KERNEL 4: RESIDUAL ADD (FFN) ---
        kernel3_add_vector_layer<<<(total_elements_embed + 255) / 256, BLOCK_SIZE>>>(
           d_residual_buffer[layer*2+1], d_model_output, d_model_output, total_elements_embed
        );cudaDeviceSynchronize();
        cudaCheckErrors("Layer 4 (Residual 2) failed");
    }
    
    // --- FINAL LAYERS ---
    printf("Running Final Output Layers...\n");
    cudaMemcpy(d_final_ln_input,d_model_output,total_elements_embed*sizeof(float),cudaMemcpyDeviceToDevice);
    // --- LAUNCH KERNEL 2: FINAL LAYER NORM ---
    kernel1_normalization_layer<<<total_tokens, BLOCK_SIZE>>>(
        d_model_output, d_ln_final_gamma, d_ln_final_beta, n_embed, 1e-5f
    );cudaDeviceSynchronize();
    cudaCheckErrors("Final LayerNorm failed");
    
    // --- LAUNCH FINAL GEMM (Unembedding) ---
    // TODO: Launch cuBLAS GEMM to get d_logits from d_model_output
    cublasSgemm(cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        VOCAB_SIZE, total_tokens, n_embed,
        &alpha,
        d_unembedding_matrix,
        VOCAB_SIZE,
        d_model_output,
        n_embed,
        &beta,
        d_logits,
        VOCAB_SIZE
    );cudaDeviceSynchronize();}
    // --- 6. Copy Results Back & Cleanup ---
    printf("Forward pass complete. Freeing memory...\n");
    printf("Backward pass unembedding...\n");
    cublasSgemm(cublas_handle,
    CUBLAS_OP_T,CUBLAS_OP_N,
    n_embed,total_tokens,VOCAB_SIZE,
&alpha,
d_unembedding_matrix, VOCAB_SIZE,
d_logits,VOCAB_SIZE,
&beta,
d_model_output_grad, n_embed);
        cudaDeviceSynchronize();
    cudaCheckErrors("Backward Unembedding failed");

    printf("Backward Pass: Final LayerNorm...\n");
    layer_norm_backward<<<total_tokens, BLOCK_SIZE>>>(d_model_output_grad,
    d_final_ln_input, d_ln_final_gamma,d_model_output_grad,n_embed,1e-5f);
    printf("Entering Decoder Reverse Loop...\n");

for (int layer = N_LAYER - 1; layer >= 0; layer--) {
    printf("Backward Pass: Layer %d FFN...\n", layer);

    // --- 1. FFN2 Backward (Contract Layer) ---
    // Formula: d_hidden = d_model_output_grad * (W_ffn2)^T
    // Dimensions: [B*S, 3072] = [B*S, 768] * [768, 3072]
    cublasSgemm(cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,     
        FFN_DIM, total_tokens, n_embed,
        &alpha,
        ffn2[layer], n_embed,         // Transpose of FFN2 weights
        d_model_output_grad, n_embed, // Incoming gradient
        &beta,
        d_ffn_hidden_grad, FFN_DIM    // Output gradient in hidden dimension
    );
    cudaDeviceSynchronize();

    // --- 2. GELU Backward ---
    // Pushes d_ffn_hidden_grad through the GELU derivative
    int grid_size_ffn = (total_elements_ffn + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gelu_backwards_kernel<<<grid_size_ffn, BLOCK_SIZE>>>(
        d_ffn_hidden_grad,    // d_in (We overwrite it in-place to save VRAM)
        d_ffn_hidden_grad,    // d_out (Incoming gradient from FFN2)
        pre_gelu_buffer[layer], // x (The saved pre-activation values)
        total_elements_ffn
    );
    cudaDeviceSynchronize();

    // --- 3. FFN1 Backward (Expand Layer) ---
    // We need a temporary buffer for the gradient coming out of FFN1 before it hits LayerNorm.
    // We can reuse a chunk of memory for this, or allocate a `d_pre_ln_grad`. 
    // Let's assume we allocated `float* d_pre_ln_grad;` (size: total_elements_embed) at the top.
    
    // Formula: d_pre_ln_grad = d_hidden_grad * (W_ffn1)^T
    cublasSgemm(cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,     
        n_embed, total_tokens, FFN_DIM,
        &alpha,
        ffn1[layer], FFN_DIM,         // Transpose of FFN1 weights
        d_ffn_hidden_grad, FFN_DIM,   // Incoming gradient from GELU
        &beta,
        d_pre_ln_grad, n_embed        // Gradient back in model dimension!
    );
    cudaDeviceSynchronize();

    // --- 4. Pre-FFN LayerNorm Backward ---
    // We use the saved residual state to get the original input to this LayerNorm
    layer_norm_backward<<<total_tokens, BLOCK_SIZE>>>(
        d_pre_ln_grad,                 // Incoming gradient from FFN1
        d_residual_buffer[layer*2+1],  // Original input (x) saved during forward pass
        d_ln_ffn_gamma[layer],         // Scale parameter
        d_pre_ln_grad,                 // Output (dx), overwriting in-place
        n_embed,
        1e-5f
    );
    cudaDeviceSynchronize();

    // --- 5. Residual Add Backward (FFN Block) ---
    // The gradient splits at a residual addition: 
    // d_model_output_grad = d_model_output_grad + d_pre_ln_grad
    kernel3_add_vector_layer<<<(total_elements_embed + 255) / 256, BLOCK_SIZE>>>(
        d_model_output_grad, d_pre_ln_grad, d_model_output_grad, total_elements_embed
    );
    cudaDeviceSynchronize();

    // ... Next up: Attention Block Backward ...
}
    // Free all memory
    cudaFree(d_input_ids);
    cudaFree(d_token_embed_matrix);
    cudaFree(d_pos_embed_matrix);
    cudaFree(d_model_output);    
     cudaFree(d_ln_final_gamma);
    cudaFree(d_ln_final_beta);
    cudaFree(d_logits);
    free(h_input_ids);
    cudaFree(d_final_ln_input);
    cudaFree(d_attn_output);
    cudaFree(d_attn_output_split);
    cublasDestroy(cublas_handle);
    
    cudaFree(d_unembedding_matrix);
    
    for(int i = 0;i<N_LAYER;i++)
    {
        cudaFree(d_layer_Wq[i]);
    cudaFree(d_layer_Wk[i]);
    cudaFree(d_layer_Wv[i]);
    cudaFree(d_layer_Wo[i]);
    cudaFree(d_V_split[i]);
    cudaFree(d_K_split[i]);
    cudaFree(d_Q_split[i]);
    cudaFree(d_Q[i]);
    cudaFree(d_K[i]);
    cudaFree(d_V[i]);
    cudaFree(ffn1[i]);
    cudaFree(ffn2[i]);
    cudaFree(ffh[i]);
    cudaFree(d_ln_attn_gamma[i]);
    cudaFree(d_ln_attn_beta[i]);
     cudaFree(d_ln_ffn_gamma[i]);
    cudaFree(d_ln_ffn_beta[i]);
        cudaFree(d_scores[i]);
    

    }
        for(int i = 0;i<N_LAYER*2;i++)
        {
                cudaFree(d_residual_buffer[i]);

        }
    
    return 0;}