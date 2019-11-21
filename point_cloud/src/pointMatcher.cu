/**
@file pointMatcher.cu
@author Taylor Nelms
*/





#include "pointMatcher.h"

float* d_A1;
float* d_A2;
float* d_O;



__global__ void multiplyNumbers(float* A1, float* A2, float* O){
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    float intermediary = A1[index] * A2[index];
    O[index] = intermediary;


}//multiplyNumbers



float testCudaFunctionality(float* arrayOne, float* arrayTwo){

    float O[32];

    cudaMalloc(&d_A1, 32);
    cudaMalloc(&d_A2, 32);
    cudaMalloc(&d_O, 32);

    dim3 threadsPerBlock(32);
    dim3 blocksPerGrid(1);


    cudaMemcpy(d_A1, arrayOne, 32 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A2, arrayTwo, 32 * sizeof(float), cudaMemcpyHostToDevice);

    multiplyNumbers<<< blocksPerGrid, threadsPerBlock >>>(d_A1, d_A2, O);

    cudaMemcpy(O, d_O, 32 * sizeof(float), cudaMemcpyDeviceToHost);

    float result = 0.0f;
    for (int i = 0; i < 32; i++){
        result += O[i]; 
    }//for

    cudaFree(d_A1);
    cudaFree(d_A1);
    cudaFree(d_O);

    return result;
    


}//testCudaFunctionality
