__global__ void kernel0_embedding_layer(const long* input_ids,const float* vector_token, const float* position_token, float* output_token,const int input_lenght,const int n_embd){
    int idx = blockIdx.x;
    int id = threadIdx.x;
    int token_pos = idx%input_lenght;
    for(int i = id;i<n_embd;i+=blockDim.x)
    {
    output_token[idx*n_embd+i] = position_token[token_pos*n_embd+i]+vector_token[input_ids[idx]*n_embd+i];
    }
}

__global__ void kernel1_normalization_layer(float* input_embd,const float* gamma, const float* beta,int n_embed,float epsilon){
    __shared__ float sh[256];
    int idx = blockIdx.x;
    int id = threadIdx.x;
    __shared__ float mean;
    __shared__ float variance;
    float sum = 0.0;
    for(int i = id;i<n_embed;i+=blockDim.x)
    {
        sum+=input_embd[idx*n_embed+i];
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
        float temp = input_embd[idx*n_embed+i]-mean;
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
        variance = sh[id]/n_embed;
    __syncthreads();
    float int_stdev = rsqrtf(variance+epsilon);
    for(int i=id;i<n_embed;i+=blockDim.x)
    {
        float x_norm = (input_embd[idx*n_embed+i]-mean)*int_stdev;
        input_embd[idx*n_embed+i] = x_norm*gamma[i]+beta[i];
    }
}

__global__ void kernel2_softmax_layer(float* input_scores,const int input_size){
    __shared__ float sh[1024];
    int idx = blockIdx.x;
    int id = threadIdx.x;
    int lr = idx%input_size;
    for(int i = id;i<input_size;i+=blockDim.x)
    {
        if(i>lr)
            sh[i]=-1e9;
        else{
            sh[i] = input_scores[idx*input_size+i]/8.0f;
        }
    }
    __syncthreads();
   __shared__ float maxs ;
   __shared__ float l_max[256];
   l_max[id]=-1e20f;
       __syncthreads();

   for(int i = id;i<input_size;i+=blockDim.x)
   {
    l_max[id] = max(sh[i],l_max[id]);
   }
   __syncthreads();
   for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (id < s) {
            l_max[id] = max(l_max[id + s],l_max[id]);
        }
        __syncthreads(); // Sync *inside* the loop
    }
    if(id==0)
        maxs = l_max[0];
            __syncthreads();

     for(int i = id;i<input_size;i+=blockDim.x)
    {
      float temp = sh[i]-maxs;
      sh[i] = expf(temp);
    }
    __syncthreads();
    __shared__ float l_sum[256];
    l_sum[id] = 0.0f;
        __syncthreads();

    for(int i = id;i<input_size;i+=blockDim.x)
   {
    l_sum[id] += sh[i];
   }
      __syncthreads();
   __shared__ float final_s;
   
   for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (id < s) {
            l_sum[id] +=l_sum[id+s];
        }
        __syncthreads(); // Sync *inside* the loop
    }
       __syncthreads();
       if(id==0)
    final_s=l_sum[0];
    __syncthreads();

     for(int i = id;i<input_size;i+=blockDim.x)
   {
        input_scores[idx*input_size+i] = sh[i]/final_s;
   }


}

__global__ void kernel3_add_vector_layer(const float* a,const float* b,float* c,const size_t N){
    size_t idx = threadIdx.x +blockDim.x*blockIdx.x;
    size_t stride = gridDim.x*blockDim.x;
    for (size_t i = idx; i < N; i += stride) {
        
        c[i] = a[i] + b[i];
    }
}

__global__ void kernel4_gelu_activation_layer(float*a,const size_t N)
{
     size_t idx = threadIdx.x +blockDim.x*blockIdx.x;
    size_t stride = gridDim.x*blockDim.x;
    for (size_t i = idx; i < N; i += stride) {
        float x = a[i];
        a[i] = 0.5 * x * (1.0 + tanhf(0.79788456 * (x + 0.044715 * x * x * x)));
    }
}

__global__ void transpose_kernel_layer(float* in,float* out,int w,int h)
{
    __shared__ float tile[32][33];
    int x = threadIdx.x+blockIdx.x*blockDim.x;
    int y = threadIdx.y+blockIdx.y*blockDim.y;
    if(x<w&&y<h)
    {
        tile[threadIdx.x][threadIdx.y]=in[y*w+x];
        
    }
    __syncthreads();
    x= blockIdx.y*blockDim.x+threadIdx.x;
    y = blockDim.y*blockIdx.x + threadIdx.y;
    if(x<h&&y<w)
        {out[y*h+x] = tile[threadIdx.y][threadIdx.x];}

}

__global__ void kernel_transpose_split_heads_layer(    const float* in,
    float* out,
const int seq_len,
const int n_heads,
const int head_dim)
{
 int token_idx = blockIdx.x;
 if(token_idx>=seq_len)return;
 int tid = threadIdx.x;
 int total_dim = n_heads*head_dim;
 for(int i = tid;i<total_dim;i+=blockDim.x)
 {
    int head_idx = i/head_dim;
    int dim_idx = i%head_dim;
    int input_idx = token_idx*total_dim+i;
    float val = in[input_idx];
    int output_idx = (head_idx*seq_len*head_dim)+(token_idx*head_dim)+dim_idx;
    out[output_idx]=val;

 }
}