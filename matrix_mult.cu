
#include <iostream>

#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers

// Matrix dimensions.


const dim3 BLOCK_DIM(16, 16);

void cpu_matrix_mult(const float* A, const float* B, float* C, int n) {
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			int partial_sum = 0;
			for(int k = 0; k < n; k++){
				partial_sum += A[i * n + k] * B[k * n + j];
			}
			C[i*n + j] = partial_sum;
    		}
  	}
}

__global__ void gpu_matrix_mult(const float* A, const float* B, float* C, int n) {
	int row = threadIdx.y + blockIdx.y * blockDim.y;
  	int col = threadIdx.x + blockIdx.x * blockDim.x;

  	if (row < n && col < n) {
  		int partial_sum = 0;
		for(int i = 0; i<n;i++){
			partial_sum += A[row * n + i] * B[i * n + col];
		}
		C[row * n + col] = partial_sum;
	}
}

__global__ void gpu_matrix_mult_tiled(const float* A, const float* B, float* C, int N){
	__shared__ int As[16*16];
	__shared__ int Bs[16*16];

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
        int ty = threadIdx.y;
        int dim = blockDim.x;
        int tmp = 0;
        for(int i = 0; i < (N / dim); i++){
                As[ty * dim + tx] = A[(row * N) + (i * dim) + tx];
                Bs[ty * dim + tx] = B[(i * dim * N) + (ty * N) + col];
                __syncthreads();
                for(int j = 0; j < dim; j++){
                        tmp += As[ty *  dim + j] * Bs[j * dim + tx];
                }
                __syncthreads();
        }

        C[row * N + col] = tmp;
}

int main(int argc, char** argv) {
  if(argc != 2){
    printf("Invalid number of arguments.\n"); 
    exit(-1);
  }

  int length = atoi(argv[1]);

  float* h_A = (float*) malloc(length*length*sizeof(float));
  float* h_B = (float*) malloc(length*length*sizeof(float));
  float* h_C = (float*) malloc(length*length*sizeof(float));
  float* h_D = (float*) malloc(length*length*sizeof(float));
  float* h_E = (float*) malloc(length*length*sizeof(float));

  srand(time(0));
  for (int row = 0; row < length; ++row) {
    for (int col = 0; col < length; ++col) {
      int idx = row * length + col;
      h_A[idx] = static_cast<float>(rand()) / RAND_MAX;
      h_B[idx] = static_cast<float>(rand()) / RAND_MAX;
    }
  }

  float *d_A, *d_B, *d_C, *d_D, *d_E;
  checkCudaErrors(cudaMalloc((void**) &d_A, length * length*sizeof(float)));
  checkCudaErrors(cudaMalloc((void**) &d_B, length * length* sizeof(float)));
  checkCudaErrors(cudaMalloc((void**) &d_C, length * length * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**) &d_D, length * length * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**) &d_E, length * length * sizeof(float)));

  checkCudaErrors(cudaMemcpy(d_A, h_A, length*length * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_B, h_B, length * length * sizeof(float), cudaMemcpyHostToDevice));

  StopWatchInterface* timer = 0;
  sdkCreateTimer(&timer);

  timer->start();
  cpu_matrix_mult(h_A, h_B, h_C, length);

  timer->stop();
  std::cout << "---- CPU Serial Time: " << timer->getTime() << " ms." << std::endl;

  const dim3 GRID_DIM((length - 1) / BLOCK_DIM.x + 1, (length - 1) / BLOCK_DIM.y + 1);

  timer->reset();
  timer->start();
  gpu_matrix_mult<<<GRID_DIM, BLOCK_DIM>>>(d_A, d_B, d_C, length);
  checkCudaErrors(cudaDeviceSynchronize());

  timer->stop();
  std::cout << "---- GPU time: " << timer->getTime() << " ms." << std::endl;
  
  timer->reset();
  timer->start();
  gpu_matrix_mult<<<GRID_DIM, BLOCK_DIM>>>(d_A, d_B, d_D, length);
  checkCudaErrors(cudaDeviceSynchronize());
  timer->stop();
  std::cout << "---- GPU Tiled time: " << timer->getTime() << " ms." << std::endl;
  checkCudaErrors(cudaMemcpy(h_A, d_C, length * length* sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_B, d_D, length * length * sizeof(float), cudaMemcpyDeviceToHost));

  float diff = 0.0f;
  float diff2 = 0.0f;
  for (int row = 0; row < length; ++row) {
    for (int col = 0; col < length; ++col) {
      int idx = row * length + col;
      diff = std::max(diff, std::abs(h_A[idx] - h_C[idx]));
      diff = std::max(diff2, std::abs(h_B[idx] - h_C[idx]));
    }
  }

  std::cout << "Max diff = " << diff  << "| Max diff 2 = "<< diff2  << std::endl;

  delete timer;

  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  return 0;
}
