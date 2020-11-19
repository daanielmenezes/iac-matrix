#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

extern "C" {
#include "timer.h"
#include "matrix_lib.h"
#include <cuda_runtime.h>
}

#define MAX_PRINT 256

typedef struct matrix Matrix;

Matrix *newMatrix( int height, int width, FILE *stream) {
    cudaError_t error;    
    int size, byteSize;
    Matrix *m = (Matrix *) malloc(sizeof(Matrix));
    if (m) {
        size = width*height;
        byteSize = size*sizeof(float);
        m->width = width;
        m->height = height;

        m->h_rows = (float *) malloc(byteSize);
        if (!m->h_rows) {
            printf("Erro de alocacao malloc\n");
            exit(1);
        }

        error = cudaMalloc(&m->d_rows, byteSize);
        if (error != cudaSuccess) {
            printf("Erro de alocacao cudaMalloc\n");
            exit(1);
        }
        
        if (stream) {
            fread(m->h_rows, sizeof(float), size, stream);
            error = cudaMemcpy(m->d_rows, m->h_rows, byteSize, cudaMemcpyHostToDevice);
            if (error != cudaSuccess) {
                printf("File: %s Line: %d. Erro cudaMemcpy: %s\n",
                        __FILE__, __LINE__, cudaGetErrorString(error));
                exit(1);
            }
        }
        else {
            memset(m->h_rows, 0, byteSize);
            error = cudaMemset(m->d_rows, 0, byteSize);
            if (error != cudaSuccess) {
                printf("Erro de cudaMemset\n");
                exit(1);
            }
        }

    }
    return m;
}

void destroyMatrix( Matrix *m ) {
    free(m->h_rows);
    cudaFree(m->d_rows);
    free(m);
}

/* write matrix values as binary file */
void writeMatrix( const Matrix *m, FILE *stream ) {
   fwrite(m->h_rows, m->width*m->height, sizeof(float), stream); 
}

/* print matrix values to stdout */
void printMatrix( const Matrix *m ){
    int i,j;
    for (i=0; i<m->height; i++) {
        for (j=0; (j<m->width) && (i*m->width+j < MAX_PRINT); j++) {
            printf("%.1f ", m->h_rows[i*m->width+j]);
            if (j == m->width-1)
                putchar('\n');
        }
    }
    if (m->width * m->height > MAX_PRINT){
        printf("\n%d limit found...skipping printing...\n", MAX_PRINT);
    }
}

FILE *openCheck(char *path, char *mode){
    FILE *f = fopen(path, mode);
    if (f == NULL){
        printf("It was not possible to open the file \"%s\": %s\n", path, strerror(errno));
        exit(1);
    }
    return f;
}

int main ( int argc, char **argv ) {
    int scalar_mult_success, matrix_matrix_success, set_grid_success;
    struct timeval start, stop, overall_t1, overall_t2;
    FILE *fileA, *fileB, *fileR1, *fileR2;
    Matrix *A, *B, *C;
    char  readMode[] = "rb";
    char writeMode[] = "wb";
    
    if (argc != 12) {
        printf("Usage: %s <float to multiply> <A matrix height> <A matrix width>"
                                            " <B matrix height> <B matrix width>"
                                            " <n of threads/block> <max n of blocks>"
                                            " <A input file> <B input file>"
                                            " <result1 file> <result2 file>\n", argv[0]);
        return 1;
    }
    gettimeofday(&overall_t1, NULL);

    fileA  = openCheck(argv[8], readMode);
    fileB  = openCheck(argv[9], readMode);
    fileR1 = openCheck(argv[10], writeMode);
    fileR2 = openCheck(argv[11], writeMode);

    set_grid_success = set_grid_size(atoi(argv[6]), atoi(argv[7]));
    if (!set_grid_success) {
        printf ("Grid size set failed. Using default values...\n");
    }

    A = newMatrix(atoi(argv[2]), atoi(argv[3]), fileA);
    B = newMatrix(atoi(argv[4]), atoi(argv[5]), fileB);
    C = newMatrix(atoi(argv[2]), atoi(argv[5]), NULL);
    if (A == NULL || B == NULL || C == NULL) {
        perror("It was not possible to allocate all the matrices");
        exit(1);
    }

    puts("--------Matrix A ---------");
    printMatrix(A);
    puts("--------Matrix B ---------");
    printMatrix(B);
    puts("--------Matrix C ---------");
    printMatrix(C);

    /* scalar product */
    printf("Executing scalar_matrix_mult(%s, A)\n", argv[1]);
    gettimeofday(&start, NULL);

    scalar_mult_success = scalar_matrix_mult(atof(argv[1]), A);
    if (!scalar_mult_success) {
        fprintf(stderr,"Error on scalar matrix multiplication\n");
        exit(1);
    }

    gettimeofday(&stop, NULL);
    printf("Scalar product time: %f ms\n", timedifference_msec(start, stop));
    printf("Writing first result to %s\n",argv[9]);
    writeMatrix(A, fileR1);
    puts("--------Matrix A ---------");
    printMatrix(A);

    /* matrix product */
    printf("Executing matrix_matrix_mult(A, B, C)\n");
    gettimeofday(&start, NULL);
    matrix_matrix_success = matrix_matrix_mult(A, B, C);
    gettimeofday(&stop, NULL);
    if (!matrix_matrix_success) {
        fprintf(stderr,"Error on matrix multiplication\n");
        exit(1);
    }
    printf("Matrix product time: %f ms\n", timedifference_msec(start, stop));
    printf("Writing second result to %s\n",argv[10]);
    writeMatrix(C, fileR2);
    puts("--------Matrix C ---------");
    printMatrix(C);

    destroyMatrix(A);
    destroyMatrix(B);
    destroyMatrix(C);

    fclose(fileA);
    fclose(fileB);
    fclose(fileR1);
    fclose(fileR2);

    gettimeofday(&overall_t2, NULL);
    printf("Overall time: %f ms\n", timedifference_msec(overall_t1, overall_t2));

    return 0;
}
