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
 */
int matrix_matrix_mult(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC) {
    /* TODO */
    return 0;
}
