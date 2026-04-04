#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <mpi.h>
#include <vector>
#include <string>

#define COMM MPI_COMM_WORLD
#define TAG_RANGE 1
#define TAG_DATA 2
#define TAG_RESULT 3
#define TAG_DIE 4
#define TAG_READY 5

__global__ void conv_kernel(float *img, float *kernel, float *out, int width,
                            int chunk_rows, int offset) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x > 0 && x < width - 1 && y < chunk_rows) {
    int local_y = y + offset;
    float sum = 0;
    for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        sum += img[(local_y + i) * width + (x + j)] * kernel[(i + 1) * 3 + (j + 1)];
      }
    }
    out[local_y * width + x] = sum;
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(COMM, &rank);
  MPI_Comm_size(COMM, &size);
  printf("Rank %d started\n", rank);

  int N = 4096; // Default
  if (argc > 1)
    N = std::stoi(argv[1]);

  if (rank == 0) {
    std::cout << "Parallel Convolution (" << N << "x" << N << ") with " << size
              << " ranks." << std::endl;
    std::cout << "Using GUIDED scheduling." << std::endl;
  }

  if (rank == 0) {
    // Scheduler
    std::vector<float> full_img(N * N, 1.0f);
    std::vector<float> full_out(N * N, 0.0f);
    int next_row = 0;
    int workers_active = size - 1;

    double start = MPI_Wtime();
    while (workers_active > 0) {
      int range[2];
      MPI_Status status;
      // We receive up to 2 ints to avoid truncation error for TAG_RESULT
      MPI_Recv(range, 2, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, COMM, &status);
      int worker = status.MPI_SOURCE;

      if (status.MPI_TAG == TAG_RESULT) {
        MPI_Recv(&full_out[range[0] * N], range[1] * N, MPI_FLOAT, worker,
                 TAG_RESULT, COMM, &status);
      }
      // If TAG_READY or after TAG_RESULT, send more work

      if (next_row < N) {
        int remaining = N - next_row;
        int chunk_size = std::max(32, remaining / (2 * (size - 1)));
        if (next_row + chunk_size > N)
          chunk_size = N - next_row;

        int range[2] = {next_row, chunk_size};
        MPI_Send(range, 2, MPI_INT, worker, TAG_RANGE, COMM);

        int send_start = std::max(0, next_row - 1);
        int send_end = std::min(N - 1, next_row + chunk_size);
        int actual_rows = send_end - send_start + 1;
        int offset = next_row - send_start;
        int header[2] = {actual_rows, offset};

        MPI_Send(header, 2, MPI_INT, worker, TAG_DATA, COMM);
        MPI_Send(&full_img[send_start * N], actual_rows * N, MPI_FLOAT, worker,
                 TAG_DATA, COMM);

        next_row += chunk_size;
      } else {
        MPI_Send(nullptr, 0, MPI_INT, worker, TAG_DIE, COMM);
        workers_active--;
      }
    }
    double end = MPI_Wtime();
    std::cout << "Time taken: " << (end - start) << std::endl;

  } else {
    // Worker
    float h_kernel[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    float *d_kernel;
    cudaMalloc(&d_kernel, 9 * sizeof(float));
    cudaMemcpy(d_kernel, h_kernel, 9 * sizeof(float), cudaMemcpyHostToDevice);

    MPI_Send(nullptr, 0, MPI_INT, 0, TAG_READY, COMM);

    float *d_img = nullptr, *d_out = nullptr;
    size_t current_gpu_size = 0;

    while (true) {
      MPI_Status status;
      int range[2];
      MPI_Recv(range, 2, MPI_INT, 0, MPI_ANY_TAG, COMM, &status);
      if (status.MPI_TAG == TAG_DIE)
        break;

      int chunk_rows = range[1];
      int header[2];
      MPI_Recv(header, 2, MPI_INT, 0, TAG_DATA, COMM, &status);
      int total_rows = header[0];
      int offset = header[1];

      std::vector<float> local_data(total_rows * N);
      MPI_Recv(local_data.data(), total_rows * N, MPI_FLOAT, 0, TAG_DATA, COMM,
               &status);
      printf("Rank %d processing rows %d to %d\n", rank, range[0],
             range[0] + range[1] - 1);

      size_t needed_size = (size_t)total_rows * N * sizeof(float);
      if (needed_size > current_gpu_size) {
        if (d_img) cudaFree(d_img);
        if (d_out) cudaFree(d_out);
        cudaMalloc(&d_img, needed_size);
        cudaMalloc(&d_out, needed_size);
        current_gpu_size = needed_size;
      }

      cudaMemcpy(d_img, local_data.data(), needed_size, cudaMemcpyHostToDevice);

      dim3 threads(16, 16);
      dim3 blocks((N + 15) / 16, (chunk_rows + 15) / 16);
      conv_kernel<<<blocks, threads>>>(d_img, d_kernel, d_out, N, chunk_rows,
                                       offset);

      cudaMemcpy(local_data.data(), d_out, needed_size, cudaMemcpyDeviceToHost);

      MPI_Send(range, 2, MPI_INT, 0, TAG_RESULT, COMM);
      MPI_Send(local_data.data() + (offset * N), chunk_rows * N, MPI_FLOAT, 0,
               TAG_RESULT, COMM);
    }
    if (d_img) cudaFree(d_img);
    if (d_out) cudaFree(d_out);
    cudaFree(d_kernel);
  }

  printf("Rank %d finished\n", rank);
  MPI_Finalize();
}