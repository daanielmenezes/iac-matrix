#include "matrix_lib.h"

/*
 *Essa função recebe um valor escalar e uma matriz como argumentos de entrada e calcula o
 *produto do valor escalar pela matriz. O resultado da operação deve ser retornado na matriz
 *de entrada. Em caso de sucesso, a função deve retornar o valor 1. Em caso de erro, a
 *função deve retornar 0.
 */
int scalar_matrix_mult(float scalar_value, struct matrix *matrix){
    int i;
    if (matrix && matrix->rows){
        for (i=0; i < matrix->width*matrix->height; i++) {
            matrix->rows[i] *= scalar_value; 
        }
        return 1;
    }
    return 0;
}

/*
 *Essa função recebe 3 matrizes como argumentos de entrada e calcula o valor do produto da
 *matriz A pela matriz B. O resultado da operação deve ser retornado na matriz C. Em caso
 *de sucesso, a função deve retornar o valor 1. Em caso de erro, a função deve retornar 0.
 *
 * OPTIMIZATION 1:
 *   operations are made in square blocks of block_size to avoid accessing long distances 
 *   between accessed memory adresses.
 */
int matrix_matrix_mult(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC) {
    int i,j,k,ib,jb,kb,block_size = 8;
    float sum;
    if (matrixA->width != matrixB->height || matrixA->height != matrixC->height || matrixB->width != matrixC->width)
        return 0;
    for (i=0; i< matrixC->height ; i+=block_size) {
        for (j=0; j<matrixC->width; j+=block_size) {
            for (k=0; k<matrixA->width; k+=block_size){
                //for each block:
                for (ib=i; ib<block_size+i; ib++){
                    for (jb=j; jb<block_size+j; jb++) {
                        sum = 0;
                        for (kb=k; kb<block_size+k; kb++) {
                            sum += matrixA->rows[ib*matrixA->width + kb] *
                                   matrixB->rows[kb*matrixB->width + jb];
                        }
                        matrixC->rows[ib*matrixC->width+jb] += sum;
                    }
                }
            }
        }
    }
    return 1;
}
