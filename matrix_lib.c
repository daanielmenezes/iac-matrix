#include <immintrin.h>
#include <pthread.h>
#include "matrix_lib.h"


static int n_threads = 1;

struct scalar_mult_arg {
    float *base;
    int elem_qty;
    int scalar_value;
};

static void *scalar_mult_worker(void *a) {
    int i;
    struct scalar_mult_arg *arg = (struct scalar_mult_arg *) a;
    __m256 temp;
    const __m256 scalar_vec = _mm256_set1_ps(arg->scalar_value);
    for (i=0; i < arg->elem_qty; i+=8) {
        temp = _mm256_load_ps(arg->base+i);
        temp = _mm256_mul_ps(scalar_vec, temp);
        _mm256_store_ps(arg->base+i, temp);
    }
    pthread_exit(NULL);
}


/*
 *Essa função recebe um valor escalar e uma matriz como argumentos de entrada e calcula o
 *produto do valor escalar pela matriz. O resultado da operação deve ser retornado na matriz
 *de entrada. Em caso de sucesso, a função deve retornar o valor 1. Em caso de erro, a
 *função deve retornar 0.
 */
int scalar_matrix_mult(float scalar_value, struct matrix *matrix){
    int i, line_qty;
    pthread_t *threads;
    struct scalar_mult_arg *args;
    if (matrix && matrix->rows){
        threads = malloc(sizeof(pthread_t) * n_threads);
        args = malloc(sizeof(struct scalar_mult_arg) * n_threads);
        line_qty = matrix->height/n_threads;
        for (i=0; i<n_threads; i++) {
            args[i].base = matrix->rows+i*line_qty;
            args[i].elem_qty = matrix->width*line_qty;
            args[i].scalar_value = scalar_value;
            pthread_create(&threads[i],NULL, scalar_mult_worker, &args[i]);
        }

        for (i=0; i<n_threads; i++) {
            if (pthread_join(threads[i], NULL))
                return 0;
        }
        free(args);
        free(threads);
        return 1;
    }
    return 0;
}




struct matrix_mult_arg {
    struct matrix *matrixA;
    struct matrix *matrixB;
    struct matrix *matrixC;
    int firstLine, lastLine;
};

static void *matrix_mult_worker(void *a) {
    int aLine, aCol, bCol, aLineIdx, bLineIdx, cLineIdx;
    struct matrix_mult_arg *arg = (struct matrix_mult_arg *) a;
    __m256 aElem, cResult, bRow;

    for (aLine = arg->firstLine; aLine < arg->lastLine; aLine++) {
        aLineIdx = aLine*arg->matrixA->width;
        cLineIdx = aLine*arg->matrixC->width;
        for (aCol=0; aCol < arg->matrixA->width; aCol++) {
            aElem = _mm256_set1_ps(arg->matrixA->rows[aLineIdx + aCol]);
            bLineIdx = aCol*arg->matrixB->width;
            for (bCol = 0; bCol < arg->matrixB->width; bCol+=8) {
                bRow = _mm256_load_ps( arg->matrixB->rows + bLineIdx + bCol);
                cResult = _mm256_load_ps( arg->matrixC->rows + cLineIdx+bCol);
                cResult = _mm256_fmadd_ps(aElem, bRow, cResult);
                _mm256_store_ps( arg->matrixC->rows + cLineIdx + bCol, cResult );
            }     
        }
    }
    pthread_exit(NULL);
}
/*
 *Essa função recebe 3 matrizes como argumentos de entrada e calcula o valor do produto da
 *matriz A pela matriz B. O resultado da operação deve ser retornado na matriz C. Em caso
 *de sucesso, a função deve retornar o valor 1. Em caso de erro, a função deve retornar 0.
 */
int matrix_matrix_mult(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC) {
    int i, line_qty, line;
    pthread_t *threads;
    struct matrix_mult_arg *args;
    if (matrixA->width != matrixB->height || matrixA->height != matrixC->height || matrixB->width != matrixC->width)
        return 0;

    threads = malloc(sizeof(pthread_t) * n_threads);
    args = malloc(sizeof(struct matrix_mult_arg) * n_threads);
    line_qty = matrixA->height/n_threads;
    for (i=0, line=0; i<n_threads; i++, line+=line_qty) {
        args[i].matrixA = matrixA;
        args[i].matrixB = matrixB;
        args[i].matrixC = matrixC;
        args[i].firstLine = line;
        args[i].lastLine = line + line_qty;
        pthread_create(&threads[i],NULL, matrix_mult_worker, &args[i]);
    }

    for (i=0; i<n_threads; i++) {
        if (pthread_join(threads[i], NULL))
            return 0;
    }
    free(args);
    free(threads);

    return 1;
}

void set_number_threads(int num_threads) {
    n_threads = num_threads;
}

