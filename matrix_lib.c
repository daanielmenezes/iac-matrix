#include <immintrin.h>
#include "matrix_lib.h"

/*
 *Essa função recebe um valor escalar e uma matriz como argumentos de entrada e calcula o
 *produto do valor escalar pela matriz. O resultado da operação deve ser retornado na matriz
 *de entrada. Em caso de sucesso, a função deve retornar o valor 1. Em caso de erro, a
 *função deve retornar 0.
 */
int scalar_matrix_mult(float scalar_value, struct matrix *matrix){
    int i;
    __m256 temp;
    const __m256 scalar_vec = _mm256_set1_ps(scalar_value);
    if (matrix && matrix->rows){
        for (i=0; i < matrix->width*matrix->height; i+=8) {
            temp = _mm256_load_ps(matrix->rows+i);
            temp = _mm256_mul_ps(scalar_vec, temp);
            _mm256_store_ps(matrix->rows+i, temp);
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
    int aLine, aCol, bCol, aLineIdx, bLineIdx, cLineIdx;
    __m256 aElem, cResult, bRow;
    if (matrixA->width != matrixB->height || matrixA->height != matrixC->height || matrixB->width != matrixC->width)
        return 0;
    for (aLine = 0; aLine < matrixA->height; aLine++) {
        aLineIdx = aLine*matrixA->width;
        cLineIdx = aLine*matrixC->width;
        for (aCol=0; aCol < matrixA->width; aCol++) {
            aElem = _mm256_set1_ps(matrixA->rows[aLineIdx + aCol]);
            bLineIdx = aCol*matrixB->width;
            for (bCol = 0; bCol < matrixB->width; bCol+=8) {
                bRow = _mm256_load_ps( matrixB->rows + bLineIdx + bCol);
                cResult = _mm256_load_ps( matrixC->rows + cLineIdx+bCol);
                cResult = _mm256_fmadd_ps(aElem, bRow, cResult);
                _mm256_store_ps( matrixC->rows + cLineIdx + bCol, cResult );
            }     
        }
    }
    return 1;
}
