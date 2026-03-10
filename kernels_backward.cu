__global__ void layer_norm_backward(
    const float* d_out,    // Incoming gradient (d_y)
    const float* input,    // Original input (x)
    const float* gamma,    // Scale parameter
    float* d_input,        // Result (dx)
    int n_embed,
    float epsilon

) {
__shared__ float sh[256];
    int idx = blockIdx.x;
    int id = threadIdx.x;
    __shared__ float mean;
    __shared__ float inv_stddev;
    float sum = 0.0;
    for(int i = id;i<n_embed;i+=blockDim.x)
    {
        sum+=input[idx*n_embed+i];
    }
    sh[id]=sum;
    __syncthreads();
for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (id < s) {
            sh[id] += sh[id + s];
        }
        __syncthreads(); // Sync *inside* the loop
    }
   if(id==0)
    mean = sh[id]/n_embed;
    __syncthreads();
     sum = 0.0;
    for(int i = id;i<n_embed;i+=blockDim.x)
    {
        float temp = input[idx*n_embed+i]-mean;
        sum+= temp*temp;
    }
    sh[id] = sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (id < s) {
            sh[id] += sh[id + s];
        }
        __syncthreads(); // Sync *inside* the loop
    }
    if(id==0)
        inv_stddev =  rsqrtf(sh[id]/n_embed+epsilon);
    __syncthreads();
    float local_sum1 = 0.0f;
    float local_sum2 = 0.0f;
     for(int i = id; i < n_embed; i += blockDim.x) {
        int pos = idx * n_embed + i;
        float val = input[pos];
        float x_hat = (val - mean) * inv_stddev;
        
        // Calculate partial gradients
        float dy_gamma = d_out[pos] * gamma[i];
        
        local_sum1 += dy_gamma;          // Sum of (dy * gamma)
        local_sum2 += dy_gamma * x_hat;  // Sum of (dy * gamma * x_hat)
    }

    sh[id] = local_sum1;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (id < s) sh[id] += sh[id + s];
        __syncthreads();
    }
    __shared__ float s1_total;
    if (id == 0) s1_total = sh[0];
    __syncthreads();

    // --- Reduction 2 (Sum2) ---
    sh[id] = local_sum2;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (id < s) sh[id] += sh[id + s];
        __syncthreads();
    }
    __shared__ float s2_total;
    if (id == 0) s2_total = sh[0];
    __syncthreads();
    float factor = inv_stddev / n_embed; // 1 / (N * sigma)

    for(int i = id; i < n_embed; i += blockDim.x) {
        int pos = idx * n_embed + i;
        float val = input[pos];
        float x_hat = (val - mean) * inv_stddev;
        
        float dy_gamma = d_out[pos] * gamma[i]; // Direct Gradient part
        
        // Formula: (1/N*sigma) * ( N*dy*gamma - sum1 - x_hat*sum2 )
        float dx = factor * ( (n_embed * dy_gamma) - s1_total - (x_hat * s2_total) );
        
        // WRITE to global memory!
        d_input[pos] = dx;
    }
}

__global__ void fused_masked_softmax_backward(
    const float* grad_output,   // Incoming gradient (dY)
    const float* softmax_output,// Original Softmax output (S)
    float* grad_input,          // Result (dX)
    const int seq_len           // 1024
)
{
    __shared__ float sum_val ;
   __shared__ float sh_sum[256];
     int idx = blockIdx.x;
    int id = threadIdx.x;
    int lr = idx%seq_len;
    float local_sum=0;
    sh_sum[id]=0;
    for(int i = id;i<seq_len;i+=blockDim.x)
    {
        local_sum+=softmax_output[idx*seq_len+i]*grad_output[idx*seq_len+i];
    }
    sh_sum[id] = local_sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (id < s) {
            sh_sum[id] = sh_sum[s+id]+sh_sum[id];
        }
        __syncthreads(); // Sync *inside* the loop
    }
    if(id==0)
        sum_val=sh_sum[0];
    __syncthreads();
    for(int i = id;i<seq_len;i+=blockDim.x)
    {
      float dx = softmax_output[idx*seq_len+i]*(grad_output[idx*seq_len+i]-sum_val)*0.125f;
      grad_input[idx*seq_len+i] = dx;
    }

}

__global__ void gelu_backwards_kernel(
    float* d_in,
    const float* d_out, 
    const float* x,
    const size_t N
){
    size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    size_t stride = blockDim.x*gridDim.x;

    for(size_t i = idx; i<N; i+=stride)
    {
        float val = x[i];
        const float sqrt_2_over_pi = 0.79788456f;
        const float coeff = 0.044715f;

        float x_cube = val*val*val;
        float inner = sqrt_2_over_pi*(val+coeff*x_cube);
        float tanh_val = tanhf(inner);
        float cdf = 0.5f*(1.0f + tanh_val);
        float d_inner = sqrt_2_over_pi*(1.0f+3.0f*coeff*val*val);
        float pdf = 0.5f*val*(1.0f-tanh_val*tanh_val)*d_inner;

        d_in[i] = d_out[i]*(cdf + pdf);

    }}

    __global__ void kernel_transpose_reverse_merge_heads_layer(const float *dK, const float*dV,const float*dQ, float *out,int seq_len,int n_embed
    ,const int head_dim, const int n_head, const int batch_size)
    {
        int idx = blockIdx.x;
        int max_idx = batch_size*seq_len*3*n_embed;
        if (idx >= max_idx) return;
        int b = idx/ (seq_len * 3 * n_embed);
        int s = (idx/(3*n_embed))%seq_len;
        int qkv_id = (idx / n_embed)%3;
        int h = (idx%n_embed)/head_dim;
        int d = idx % head_dim;
        int input_idx = (b*n_head*seq_len*head_dim)+(h*seq_len*head_dim)+(s*head_dim) + d;
        if (qkv_id == 0) {
    out[idx] = dQ[input_idx];
} else if (qkv_id == 1) {
    out[idx] = dK[input_idx];
} else {
    out[idx] = dV[input_idx];
}
    }

__global__ void bias_layer(float * in, float *out,const int batch_size, const int seq_len, const int n_embed)
{
    size_t idx = threadIdx.x +blockDim.x*blockIdx.x;
    size_t stride = n_embed;
    if (idx >= n_embed) return;
    int N = batch_size*seq_len*n_embed;
    float local_sum = 0.0f;
    for (size_t i = idx; i < N; i += stride) {
        
       local_sum+=in[i];
    }
    out[idx] = local_sum;
} 

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
){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for(size_t i = idx; i<N;i+=stride)
    {
        float p = params[i];
        float g = grads[i];
        float m = m_memory[i];
        float v = v_memory[i];
         m = beta1*m + (1-beta1)*g;
         v = beta2*v +(1-beta2)*g*g;
         float b_m = m/bias_correction_1;
         float b_v = v/bias_correction_2;
        float step = b_m/(sqrtf(b_v)+eps)+weight_decay*p;
        p = p-(lr*step);
        params[i] = p;
        m_memory[i] = m;
        v_memory[i] = v;
    }

}

__global__ void zero_grad_kernel(float * in,const int size)
{
 size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
 if(idx>=size)
    return;
  in[idx] = 0;
}