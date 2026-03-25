#include <mpi.h>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdio>

#define COMM MPI_COMM_WORLD

__global__
void conv_kernel(
    unsigned char* input,
    unsigned char* output,
    int width,
    int rows
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x > 0 && x < width - 1 && y > 1 && y < rows){
        int idx = y * width + x;

        int sum =
            input[idx - width] +
            input[idx + width] +
            input[idx - 1] +
            input[idx + 1];

        output[idx] = sum / 4;
    }
}

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(COMM, &rank);
    MPI_Comm_size(COMM, &size);
    
    std::cout << "Rank " << rank << " started\n";

    if(argc < 3){
        if(rank == 0){
            std::cerr << "Usage: program input output" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    cv::Mat image;
    unsigned char* image_data = nullptr;

    int width, height;

    if(rank == 0){
        image = cv::imread(argv[1],cv::IMREAD_GRAYSCALE);
        
        if(image.empty()){
            std::cerr << "Failed to load image" << std::endl;
            MPI_Abort(COMM, 1);
        }

        width = image.cols;
        height = image.rows;
        image_data = image.data;
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, COMM);
    MPI_Bcast(&height, 1, MPI_INT, 0, COMM);

    int rows = height / size;

    std::vector<unsigned char> local((rows + 2) * width);
    std::vector<unsigned char> result((rows + 2) * width);

    MPI_Barrier(COMM);
    auto start = std::chrono::high_resolution_clock::now();

    MPI_Scatter(
        image_data,
        rows * width,
        MPI_UNSIGNED_CHAR,
        local.data() + width,
        rows * width,
        MPI_UNSIGNED_CHAR,
        0,
        COMM
    );

    unsigned char *d_input, *d_output;

    cudaMalloc(&d_input, rows * width * sizeof(unsigned char));
    cudaMalloc(&d_output, rows * width * sizeof(unsigned char));

    cudaMemcpy(
        d_input,
        local.data(),
        rows * width * sizeof(unsigned char),
        cudaMemcpyHostToDevice
    );

    dim3 threads(16,16);
    dim3 blocks(
        (width + 15)/16,
        (rows + 15)/16
    );

    cudaMemset(d_output, 0, rows * width * sizeof(unsigned char));

    conv_kernel<<<blocks, threads>>>(
        d_input,
        d_output,
        width,
        rows
    );

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();

    cudaMemcpy(
        result.data(),
        d_output,
        rows * width * sizeof(unsigned char),
        cudaMemcpyDeviceToHost
    );

    cudaFree(d_input);
    cudaFree(d_output);

    MPI_Gather(
        result.data(),
        rows * width,
        MPI_UNSIGNED_CHAR,
        image_data,
        rows * width,
        MPI_UNSIGNED_CHAR,
        0,
        COMM
    );

    MPI_Barrier(COMM);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count();

    double max_time;

    MPI_Reduce(
        &elapsed,
        &max_time,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );

    if(rank == 0){
        cv::imwrite(argv[2],image);
        std::cout<< "Time Taken "<< max_time<< std::endl;
    }

    std::cout << "Finished rank " << rank << std::endl;

    MPI_Finalize();
}