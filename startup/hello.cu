#include <cstdio>

// The kernel: runs on the GPU, one thread per array element
__global__ void add(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's global index
    if (i < n) x[i] += 1.0f;                        // guard against extra threads
}

int main() {
    int n = 1 << 26;                     // about 1 million elements
    float* x;
    cudaMalloc(&x, n * sizeof(float));   // allocate memory on the GPU

    // Launch enough 256-thread blocks to cover all n elements
    for (int r = 0; r < 10; ++r) {
        add<<<(n + 255) / 256, 256>>>(x, n);
    }

    cudaDeviceSynchronize();             // wait for the GPU to finish
    printf("done\n");
    cudaFree(x);
}