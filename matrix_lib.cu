extern "C" {
#include "matrix_lib.h"
}

#define  DEFAULT_TPB  256
#define  DEFAULT_MAX_BPG 4096

static int threads_per_block   = DEFAULT_TPB;
static int max_blocks_per_grid = DEFAULT_MAX_BPG; 

__global__
void scalar_mult_kernel(int scalar, int size,  float *d_rows) {
    int i;

    for (i = blockIdx.x * blockDim.x + threadIdx.x;
         i < size;
         i += blockDim.x * gridDim.x) 
    {
        d_rows[i] *= scalar; 
    }
}


/*
 *Essa função recebe um valor escalar e uma matriz como argumentos de entrada e calcula o *produto do valor escalar pela matriz. O resultado da operação deve ser retornado na matriz
 *de entrada. Em caso de sucesso, a função deve retornar o valor 1. Em caso de erro, a
 *função deve retornar 0.
 */
int scalar_matrix_mult(float scalar_value, struct matrix *matrix){
    int size, numBlocks;
    cudaError_t error;
    if (matrix && matrix->h_rows && matrix->d_rows){
        size = matrix->width * matrix->height;


        numBlocks = (size + threads_per_block - 1) / threads_per_block;
        if (numBlocks > max_blocks_per_grid)
            numBlocks = max_blocks_per_grid;


        scalar_mult_kernel<<<numBlocks, threads_per_block>>>(scalar_value, size, matrix->d_rows);
        cudaDeviceSynchronize();

        error = cudaMemcpy(matrix->h_rows,  matrix->d_rows, size*sizeof(float), cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            return 0;
        }
        
        return 1;
    }
    return 0;
}

/*
wa == hb
wc == wb
hc == ha
*/
__global__
void matrix_mult_kernel(float *d_a, float *d_b, float *d_c, int wa, int ha, int wb) {
    int i;

    for (i = blockIdx.x * blockDim.x + threadIdx.x;
         i < wb*ha;
         i += blockDim.x * gridDim.x) 
    {
        int row = i/wb;
        int col = i%wb;
        float temp = 0;
        for (int k=0; k<wa; k++) {
            temp += d_a[row*wa + k] * d_b[k*wb + col];
        }
        d_c[i] = temp;
    }
}

/*
 *Essa função recebe 3 matrizes como argumentos de entrada e calcula o valor do produto da
 *matriz A pela matriz B. O resultado da operação deve ser retornado na matriz C. Em caso
 *de sucesso, a função deve retornar o valor 1. Em caso de erro, a função deve retornar 0.
 */
int matrix_matrix_mult(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC) {
    int size, numBlocks;
    cudaError_t error;
    if (matrixA->width != matrixB->height || matrixA->height != matrixC->height || matrixB->width != matrixC->width)
        return 0;
    
    size = matrixB->width * matrixA->height;


    numBlocks = (size + threads_per_block - 1) / threads_per_block;
    if (numBlocks > max_blocks_per_grid)
        numBlocks = max_blocks_per_grid;

    matrix_mult_kernel<<<numBlocks, threads_per_block>>>(matrixA->d_rows, matrixB->d_rows, matrixC->d_rows, matrixA->width, matrixA->height, matrixB->width);
    cudaDeviceSynchronize();

    error = cudaMemcpy(matrixC->h_rows, matrixC->d_rows, size*sizeof(float), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        return 0;
    }
    
    return 1;
}

int set_grid_size(int n_threads_per_block, int n_max_block_per_grid) {
    if (n_threads_per_block <= 1024  &&  n_max_block_per_grid <= 65535) {
        threads_per_block = n_threads_per_block;
        max_blocks_per_grid = n_threads_per_block;
        return 1;
    }
    threads_per_block   = DEFAULT_TPB;
    max_blocks_per_grid = DEFAULT_MAX_BPG;
    return 0;
}
