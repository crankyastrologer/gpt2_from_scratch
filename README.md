Hello Everyone this is my readme for the project GPT2 from scratch using cuda
In this is try to implement gpt2 using c++ and cuda including forward pass backward pass training and in the end hopefully deployment
One question that comes to mind is why create from scratch when you can use it by just 1 line python code?
One of the main reason for doing this is I love working with generative AI and even in my current role i am working on differnt LLM agents trying to solve different problems
Even though I regularly read paper and have gone through the whole transformer architecture I always had a knagging feeling that i don't understand it to the depth I will like 
I mean yeah i can tell you what a decoder how the maths of attention works but somehow it always felt like surface level knowledge 
To remedy this i followed some youtube videos and build one but still to no avail
Hence i have started this project to make transformer from the lowest practical level using cuda and c++
This document i will treat like a journal to notedown what i have done what part i am stuck on 
This will also serve as documentation where i will high light what each variable and function does in great detail as to make sure i am understanding what i am making 
In the I hope i will understand what might be one of the most imp invention for the next few decades work

# Brief about parallelism and efficent cuda operations 
So, as we all know GPU is used for it's unmatched parallel processing capacity, if your process requires a bunch of comparitively simple operations that can be done without affecting each other ie parallely, GPU is your best bet. But that too comes with it's caveats. Without going into too much details basically gpu can't command all the threads simuntaneously ie at one clock or cycle the SM(Streaming Multiprocessor) can only allocate task to 1 warp usually around 32 threads and then these threads take their sweet time to complete their tasks meanwhile SM allocates tasks to other warps. so it is more like how async work in python or javascript if replace SM with a thread and warps with IO operations. So we have to make sure SM always have threads to allocate so we should always try to intialize atleast as many threads as are present in a gpu ideally much more if it helps our tasks.

Second thing is latency of different operations. Not all operations take the same amount of time and one of the most expensive operations is memory calls which could be many times slower than arithmatic operations but the thread can only work on the things that are present in the cache, So we necessarily needs to carry out memory operations to get the data in cache for nearly every operation. Hence, we need to minimize our calls to the memory. Thankfully GPU helps us in this task basically as we discussed earlier all threads in a warp carry out the same instruction that also includes reading from memory. So for eg if threads need the values from the memory in a contiguous fashion( for eg if thread 0 needs 0th element from from an array in memory and thread 1 needs 1st element) these gets clubbed into one large memory request so instead of N memory request each with some latency attached we just need 1 request drastically reducing time taken

So from this the 2 most imp principles for utilizing gpu parallelism to the max is to:
- Intialize and use as many threads as possible.
- Try to make contiguous memory requests where ever possible 


So in all of our kernels we will try to follow these principles hence if wonder why we are using relatively complicated operations when same can be done with simpler ones it is most probably because of this.
<br>
## Grid, Block, Thread, and Warp
- **Grid**: A grid is basically a collection of blocks you intiate perform a task. It is a GPU wide structure.
- **Block**: A block is a software construct it is basically a collection of threads that have access to the same shared memory so can share info between them efficiently. A block remains with one SM till the end of it's execution. 
- **Threads**: Threads are the basic unit of execution they perform the unit tasks like addition, subtraction etc. Each thread also have some local memory called register.
- **Warps**: An SM can only allocate one task in a cycle so to allocate tasks to 1024 threads it would require 1024 cycles, to make this process more efficient we club the threads into sets of 32 that execute together these are called warps with this we would only require 32 cycles to allocate tasks to 1024 threads.

Most of the code in these kernels will execute Grids of more than 1 block with a block size of 256 this is because:
1. Nvidia only supports 1024 threads per block so to intialize more than that you kinda need more blocks.
2. A block cannot be split between SMs so if you just create 1 huge block only 1 SM will be working.
3. Shared memory that can be provided to a block is limited so you can run into an issue where the memory on the chip is not enough for the block.

Hence due to this we intialize a grid of n blocks each with m no of threads.<br>



This section will have explanation for each kernel i have used and why they are necessary 
let's start in the order i created them 

```
__global__ void kernel0_embedding_layer(const long* input_ids,const float* vector_token, const float* position_token, float* output_token,const int input_lenght,const int n_embd){
    int idx = blockIdx.x;
    int id = threadIdx.x;
    int token_pos = idx%input_lenght;
    for(int i = id;i<n_embd;i+=blockDim.x)
    {
    output_token[idx*n_embd+i] = position_token[token_pos*n_embd+i]+vector_token[input_ids[idx]*n_embd+i];
    }
}
```
This is a comparitively simple funciton <br>
input_ids: Array of input words of 1024 length with padding assumed to be tokenized<br>
position_token: To add position info to the embeddings not sure which one i will use yet<br>
vector_token: this is the main vectorizer these embeddings store the semantic info about the words.<br>
n_embd: This is the dimensions of our vector embeddings here it is 768<br>
<br>
So to follow those principles we are doing all these things.
1. first we are intializing 1024 blocks each with 256 threads so about 262144 threads.(blocks can be seen as a group of threads that have a shared memory it resides on the same die is cache but we can choose what goes here ) <br>
2. So basically what happens is this code runs on all threads of a warp at the same time and as you can see from the for loop each threads call to memory is linked to the thread id and is offset by 1 so it is basically a contigous call and will work as single memory request.<br>

```
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
```
 So this kernel is called a lot in the code and it's function as the name suggests is to normalize the outputs to keep them in a manageable range:
 This kernel works in 3 distinct steps:
 1. **Calculating the sum**:As name suggests in this step we calculate the sum of all values in the dimensions of each token so a single block is assigned to each token. For this first every thread is assigned an index whose data it loads in the shared memory. But as we only have 256 threads and around 768 elements so we first do a partial sum each thread with help of a for loop adds to it's sum it's element and the elements and wait for other elements to do the same. now in next iteration all the threads again ask for the value of next 256 elements and add thier corresponding element to their sum. This helps us keep the shared memory size limited to 256.<br>
 Now we got an array we need to sum the naive approach would be to make each thread add their part to the shared sum variable, but this will even with all the threads will become a synchornous operation as only one thread can access the variable at a time and others threads would have to wait.
 Hence, we will do a parallel reduction here in which we will use a for loop in which at each point the thread will add it's value and the value in the thread remaining_values-id. Using we will be able to calculate the sum of the array without causing a memory access bottleneck.
 2. **Calculating the variance**: Variance is calculated in a similar fashion first subtract mean from indivdual array elements and then square them before finally using parallel reduction to add them altogether.
 3. **Normalising the input**: Here we simply normalize our inputs by removing mean and dividing by standard diviation(epsilon is added to make sure there are no divide by zero erros if variance is 0 for some reason).

``` 
__global__ void kernel3_add_vector_layer(const float* a,const float* b,float* c,const size_t N){
    size_t idx = threadIdx.x +blockDim.x*blockIdx.x;
    size_t stride = gridDim.x*blockDim.x;
    for (size_t i = idx; i < N; i += stride) {
        
        c[i] = a[i] + b[i];
    }
} 
```
This is the simplest kernel here we are performing simple vector addition by assigning each element a thread and loop over the array until all elements are added.
```

__global__ void kernel4_gelu_activation_layer(float*a,const size_t N)
{
     size_t idx = threadIdx.x +blockDim.x*blockIdx.x;
    size_t stride = gridDim.x*blockDim.x;
    for (size_t i = idx; i < N; i += stride) {
        float x = a[i];
        a[i] = 0.5 * x * (1.0 + tanhf(0.79788456 * (x + 0.044715 * x * x * x)));
    }
}
```
This layer is similar to the add vector layer but instead of adding 2 vectors we loop over the vector and apply Gelu activation function to each element of the matrix