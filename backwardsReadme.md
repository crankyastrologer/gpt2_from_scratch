Hello this readme file will exclusively deal with the backward pass of the kernel. The plan was to put it all in a single file but with how big it has already gotten it doesn't make sense.
layer
```c
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
```