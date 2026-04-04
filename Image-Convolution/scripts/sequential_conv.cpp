#include <iostream>
#include <vector>
#include <chrono>

#define N 4096

float image[N][N];

float kernel[3][3] = {
    {0, -1, 0},
    {-1, 5, -1},
    {0, -1, 0}
};

float output[N][N];

int main(){
    auto start = std::chrono::high_resolution_clock::now();

    for(int i = 1; i < N - 1; i++){
        for(int j = 1; j < N - 1; j++){
            float sum = 0;

            for(int ki = -1; ki <= 1; ki++){
                for(int kj = -1; kj <= 1; kj++){
                    sum += image[i + ki][j + kj] * kernel[ki + 1][kj + 1];
                }
            }

            output[i][j] = sum;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    double time = std::chrono::duration<double>(end - start).count();

    std::cout<<"Time taken: "<<time<<std::endl;
}