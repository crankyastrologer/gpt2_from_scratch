#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include "kernels.h"
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
    printf("starting gpt2");
    int batch_size = 1;
    int seq_len = SEQ_LEN;
    int n_embed = N_EMBED;
    int head_dim = HEAD_DIM;
    size_t total_tokens = batch_size * seq_len;
    size_t total_elements_embed = total_tokens * n_embed;
    size_t total_elements_ffn = total_tokens * FFN_DIM;   
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
    float* d_residual_buffer; // For residual connections: [B*S, E]
    cudaMalloc(&d_residual_buffer, total_elements_embed * sizeof(float));
    
    // (Mock) LayerNorm Weights
    float* d_ln_gamma;
    float* d_ln_beta;
    cudaMalloc(&d_ln_gamma, n_embed * sizeof(float));
    cudaMalloc(&d_ln_beta, n_embed * sizeof(float));

    // (Mock) Output
    float* d_logits;
    cudaMalloc(&d_logits, total_tokens * VOCAB_SIZE * sizeof(float));
    
    printf("Memory allocated...\n");

    // --- 4. Initialize Mock Data (Example) ---
    for(int i = 0; i < total_tokens; i++) h_input_ids[i] = 1; // Just use token "1"
    cudaMemcpy(d_input_ids, h_input_ids, total_tokens * sizeof(long), cudaMemcpyHostToDevice);
    // (In real life, we'd cudaMemcpy all the pre-trained weights)

    // --- 5. EXECUTE THE FORWARD PASS ---

    // --- LAUNCH KERNEL 1: FUSED EMBEDDING (Layer 0) ---
    printf("Running Layer 0: Embedding...\n");
    kernel0_embedding_layer<<<total_tokens, BLOCK_SIZE>>>(
        d_input_ids, d_token_embed_matrix, d_pos_embed_matrix, 
        d_model_output, seq_len, n_embed
    );cudaDeviceSynchronize();
    cudaCheckErrors("Layer 0 Embedding failed");
float *d_layer_Wq;
        cudaMalloc(&d_layer_Wq,n_embed*n_embed*sizeof(float));
        float  *d_layer_Wk, *d_layer_Wv, *d_layer_Wo;
    cudaMalloc(&d_layer_Wk, n_embed * n_embed * sizeof(float));
    cudaMalloc(&d_layer_Wv, n_embed * n_embed * sizeof(float));
    cudaMalloc(&d_layer_Wo, n_embed * n_embed * sizeof(float));
        float *d_Q, *d_K, *d_V;
    cudaMalloc(&d_Q, total_tokens * n_embed * sizeof(float));
    cudaMalloc(&d_K, total_tokens * n_embed * sizeof(float));
    cudaMalloc(&d_V, total_tokens * n_embed * sizeof(float));
        float *d_Q_split, *d_K_split, *d_V_split;
    cudaMalloc(&d_Q_split, total_tokens * n_embed * sizeof(float));
    cudaMalloc(&d_K_split, total_tokens * n_embed * sizeof(float));
    cudaMalloc(&d_V_split, total_tokens * n_embed * sizeof(float));
        cudaMalloc(&d_model_output,total_tokens*n_embed*sizeof(float));
         float *d_scores;
size_t scores_size = (size_t)batch_size * N_HEAD * seq_len * seq_len;
cudaMalloc(&d_scores, scores_size * sizeof(float));    float *d_attn_output;
    cudaMalloc(&d_attn_output, total_tokens * n_embed * sizeof(float));
        float *d_attn_output_split;
    cudaMalloc(&d_attn_output_split, total_tokens * n_embed * sizeof(float));
    float *ffn1,*ffn2,*ffh;
    cudaMalloc(&ffn1,n_embed*FFN_DIM*sizeof(float));
    cudaMalloc(&ffn2,FFN_DIM*n_embed*sizeof(float));
    cudaMalloc(&ffh,total_tokens*FFN_DIM*sizeof(float));
    float* d_unembedding_matrix;
    cudaMalloc(&d_unembedding_matrix, n_embed * VOCAB_SIZE * sizeof(float));
    float alpha = 1.0f;
        float beta = 0.0f;
    printf("Memory allocated...\n");

// --- DECODER LOOP ---
    for (int layer = 0; layer < N_LAYER; layer++) {
        printf("Running Decoder Layer %d...\n", layer);
        
        // Save for residual: d_residual_buffer = d_model_output
        cudaMemcpy(d_residual_buffer, d_model_output, total_elements_embed * sizeof(float), cudaMemcpyDeviceToDevice);
        
        // --- LAUNCH KERNEL 2: LAYER NORM (Pre-Attention) ---
        kernel1_normalization_layer<<<total_tokens, BLOCK_SIZE>>>(
            d_model_output, d_ln_gamma, d_ln_beta, n_embed, 1e-5f
        );cudaDeviceSynchronize();
        
        cudaCheckErrors("Layer 1 (Pre-Attention) LayerNorm failed");

        // --- ATTENTION BLOCK ---
        // TODO: Launch cuBLAS GEMM calls for Q, K, V
        
        cublasSgemm(cublas_handle,CUBLAS_OP_N,CUBLAS_OP_N,
            n_embed,
            total_tokens,
            n_embed,
            &alpha,
            d_layer_Wq,
            n_embed,
            d_model_output,
            n_embed,
            &beta,
            d_Q,
            n_embed
        
        );cudaDeviceSynchronize();
        cublasSgemm(cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n_embed, total_tokens, n_embed,
            &alpha,
            d_layer_Wk,    // Weights for K
            n_embed,
            d_model_output, // Same input
            n_embed,
            &beta,
            d_K,           // K Result buffer
            n_embed
        );cudaDeviceSynchronize();

        // V
        cublasSgemm(cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n_embed, total_tokens, n_embed,
            &alpha,
            d_layer_Wv,    // Weights for V
            n_embed,
            d_model_output, // Same input
            n_embed,
            &beta,
            d_V,           // V Result buffer
            n_embed
        );cudaDeviceSynchronize();
        cublasSgemm(cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            total_tokens,  // n
            total_tokens,  // m
            n_embed,       // k
            &alpha,
            d_K,           // B (K)
            n_embed,       // ldb
            d_Q,           // A (Q)
            n_embed,       // lda
            &beta,
            d_scores,      // C
            total_tokens   // ldc
        );cudaDeviceSynchronize();
        // TODO: Reshape for Multi-Head
        // TODO: Launch batched cuBLAS GEMM for Q*K^T (Scores)
        int grid_trans = total_tokens;
        kernel_transpose_split_heads_layer<<<grid_trans, BLOCK_SIZE>>>(d_Q, d_Q_split, seq_len, N_HEAD, head_dim);
        kernel_transpose_split_heads_layer<<<grid_trans, BLOCK_SIZE>>>(d_K, d_K_split, seq_len, N_HEAD, head_dim);
        kernel_transpose_split_heads_layer<<<grid_trans, BLOCK_SIZE>>>(d_V, d_V_split, seq_len, N_HEAD, head_dim);
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
            d_K_split,head_dim, stride_K,
            d_Q_split, head_dim, stride_Q,
            &beta,
            d_scores, seq_len, stride_Scores,
            N_HEAD


        );
        cudaCheckErrors("Batched Q*K^T failed");
        
        // --- LAUNCH KERNEL 3: FUSED MASKED SOFTMAX ---
        // (This assumes a 'd_scores' buffer was calculated and has shape [B*H*S, S])
        // int grid_size_softmax = total_tokens * N_HEAD;
        // fused_masked_softmax<<<grid_size_softmax, BLOCK_SIZE>>>(d_scores, seq_len);
        kernel2_softmax_layer<<<N_HEAD * seq_len, BLOCK_SIZE>>>(d_scores,seq_len);cudaDeviceSynchronize();
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
        d_V_split, head_dim, stride_V,
        d_scores, seq_len, stride_Scores,
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
            d_layer_Wo,    // Output Projection Weights
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
            d_residual_buffer, d_model_output, d_model_output, total_elements_embed
        );cudaDeviceSynchronize();
        cudaCheckErrors("Layer 4 (Residual 1) failed");

        // --- FFN BLOCK ---
        // Save for residual: d_residual_buffer = d_model_output
        cudaMemcpy(d_residual_buffer, d_model_output, total_elements_embed * sizeof(float), cudaMemcpyDeviceToDevice);
        

        // --- LAUNCH KERNEL 2: LAYER NORM (Pre-FFN) ---
        kernel1_normalization_layer<<<total_tokens, BLOCK_SIZE>>>(
            d_model_output, d_ln_gamma, d_ln_beta, n_embed, 1e-5f
        );cudaDeviceSynchronize();
        cudaCheckErrors("Layer 1 (Pre-FFN) LayerNorm failed");
        
        // TODO: Launch cuBLAS GEMM for FFN Expand (d_model_output -> d_ffn_hidden)
        cublasSgemm(cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            FFN_DIM, total_tokens, n_embed,
            &alpha,
            ffn1,
            FFN_DIM,
            d_model_output,
            n_embed,
            &beta,
            ffh,
            FFN_DIM
        );cudaDeviceSynchronize();
        
        // --- LAUNCH KERNEL 5: GELU ---
        // (This assumes d_ffn_hidden is [B*S, FFN_DIM])
        // gelu_kernel<<<(total_elements_ffn + 255) / 256, BLOCK_SIZE>>>(d_ffn_hidden, total_elements_ffn);
        int grid_size_ffn = (total_elements_ffn+BLOCK_SIZE-1)/BLOCK_SIZE;
        kernel4_gelu_activation_layer<<<grid_size_ffn, BLOCK_SIZE>>>(ffh,total_elements_ffn);
        cudaCheckErrors("GELU failed");
        // TODO: Launch cuBLAS GEMM for FFN Contract (d_ffn_hidden -> d_model_output)
        cublasSgemm(cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n_embed, total_tokens, FFN_DIM,
            &alpha,
            ffn2, // d_layer_ffn2_weight
            n_embed,
            ffh, // d_ffn_hidden
            FFN_DIM,
            &beta,
            d_model_output, // Write back to main buffer
            n_embed
        );cudaDeviceSynchronize();
        // --- LAUNCH KERNEL 4: RESIDUAL ADD (FFN) ---
        kernel3_add_vector_layer<<<(total_elements_embed + 255) / 256, BLOCK_SIZE>>>(
            d_residual_buffer, d_model_output, d_model_output, total_elements_embed
        );cudaDeviceSynchronize();
        cudaCheckErrors("Layer 4 (Residual 2) failed");
    }
    
    // --- FINAL LAYERS ---
    printf("Running Final Output Layers...\n");

    // --- LAUNCH KERNEL 2: FINAL LAYER NORM ---
    kernel1_normalization_layer<<<total_tokens, BLOCK_SIZE>>>(
        d_model_output, d_ln_gamma, d_ln_beta, n_embed, 1e-5f
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
    );cudaDeviceSynchronize();
    // --- 6. Copy Results Back & Cleanup ---
    printf("Forward pass complete. Freeing memory...\n");

    // Free all memory
    cudaFree(d_input_ids);
    cudaFree(d_token_embed_matrix);
    cudaFree(d_pos_embed_matrix);
    cudaFree(d_model_output);
    cudaFree(d_residual_buffer);
    cudaFree(d_ln_gamma);
    cudaFree(d_ln_beta);
    cudaFree(d_logits);
    free(h_input_ids);
    cudaFree(d_layer_Wq);
    cudaFree(d_layer_Wk);
    cudaFree(d_layer_Wv);
    cudaFree(d_layer_Wo);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_scores);
    cudaFree(d_attn_output);
    cublasDestroy(cublas_handle);
    cudaFree(ffn1);
    cudaFree(ffn2);
    cudaFree(ffh);
    cudaFree(d_unembedding_matrix);
    
    return 0;}