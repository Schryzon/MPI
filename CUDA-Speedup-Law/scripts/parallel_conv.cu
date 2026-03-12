#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>

#define N 4096
#define COMM MPI_COMM_WORLD

__global__
void conv_kernel(float *img, float *kernel, float *out, int width){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x > 0 && x < width - 1 && y > 0 && y < width - 1){
        float sum = 0;

        for(int i = -1; i <= 1; i++){
            for(int j = -1; j <= 1; j++){
                sum += img[(x + i) * width + (y + j)] * kernel[(i + 1) * 3 + (j + 1)];
            }
        }

        out[x * width + y] = sum;
    }
}


int main(int argc, char** argv){
    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(COMM, &rank);
    MPI_Comm_size(COMM, &size);

    int rows = N / size;

    float *chunk = new float[rows * N];
    float *result = new float[rows * N];

    float *d_img, *d_out, *d_kernel;

    cudaMalloc(&d_img, rows * N * sizeof(float));
    cudaMalloc(&d_out, rows * N * sizeof(float));
    cudaMalloc(&d_kernel, 9 * sizeof(float));

    MPI_Barrier(COMM);

    double start = MPI_Wtime();

    cudaMemcpy(d_img, chunk, rows * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((rows + 15)/16, (N+15)/16);

    conv_kernel<<<blocks, threads>>>(d_img, d_kernel, d_out, N);

    cudaMemcpy(result, d_out, rows * N * sizeof(float), cudaMemcpyDeviceToHost);

    MPI_Barrier(COMM);

    double end = MPI_Wtime();
    double local_time = end - start;
    double total_time;

    MPI_Reduce(&local_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, COMM);

    if(rank == 0){
        std::cout<<"Time taken: "<<total_time<<std::endl;
    }

    MPI_Finalize();
}