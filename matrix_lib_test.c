/*
 *Crie um programa em linguagem C, chamado matrix_lib_test.c, que implemente um código para
 *testar a biblioteca matrix_lib.c. Esse programa deve receber um valor escalar float, a dimensão da
 *primeira matriz (A), a dimensão da segunda matriz (B) e o nome de quatro arquivos binários de
 *floats na linha de comando de execução. O programa deve inicializar as duas matrizes (A e B)
 *respectivamente a partir dos dois primeiros arquivos binários de floats e uma terceira matriz (C)
 *com zeros. A função scalar_matrix_mult deve ser chamada com os seguintes argumentos: o valor
 *escalar fornecido e a primeira matriz (A). O resultado (retornado na matriz A) deve ser
 *armazenado em um arquivo binário usando o nome do terceiro arquivo de floats. Depois, a função
 *matrix_matrix_mult deve ser chamada com os seguintes argumentos: a matriz A resultante da
 *função scalar_matrix_mult, a segunda matriz (B) e a terceira matriz (C). O resultado (retornado na
 *matriz C) deve ser armazenado com o nome do quarto arquivo de floats.
 *Exemplo de linha de comando:
 *matrix_lib_test 5.0 8 16 16 8 floats_256_2.0f.dat floats_256_5.0f.dat result1.dat result2.dat
 *Onde,
 *5.0 é o valor escalar que multiplicará a primeira matriz;
 *8 é o número de linhas da primeira matriz;
 *16 é o número de colunas da primeira matriz;
 *16 é o número de linhas da segunda matriz;
 *8 é o número de colunas da segunda matriz;
 *floats_256_2.0f.dat é o nome do arquivo de floats que será usado para carregar a primeira matriz;
 *floats_256_5.0f.dat é o nome do arquivo de floats que será usado para carregar a segunda matriz;
 *result1.dat é o nome do arquivo de floats onde o primeiro resultado será armazenado;
 *result2.dat é o nome do arquivo de floats onde o segundo resultado será armazenado.
 *O programa principal deve cronometrar o tempo de execução geral do programa (overall time) e o
 *tempo de execução das funções scalar_matrix_mult e matrix_matrix_mult. Para marcar o início e o
 *final do tempo em cada uma das situações, deve-se usar a função padrão gettimeofday disponível
 *em <sys/time.h>. Essa função trabalha com a estrutura de dados struct timeval definida em
 * <sys/time.h>. 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "timer.h"
#include "matrix_lib.h"

#define MAX_PRINT 256

typedef struct matrix Matrix;

Matrix *newMatrix( int height, int width, FILE *stream) {
    Matrix *m = malloc(sizeof(Matrix));
    if (m) {
        m->width = width;
        m->height = height;
        m->rows = aligned_alloc(32, width*height*sizeof(float));
        if (m->rows && stream) {
            fread(m->rows, sizeof(float), width*height, stream);
        }
        else if (!m->rows) {
            free(m);
            m=NULL;
        }
    }
    return m;
}

void destroyMatrix( Matrix *m ) {
    free(m->rows);
    free(m);
}

/* write matrix values as binary file */
void writeMatrix( const Matrix *m, FILE *stream ) {
   fwrite(m->rows, m->width*m->height, sizeof(float), stream); 
}

/* print matrix values to stdout */
void printMatrix( const Matrix *m ){
    int i,j;
    for (i=0; i<m->height; i++) {
        for (j=0; (j<m->width) && (i*m->width+j < MAX_PRINT); j++) {
            printf("%.1f ", m->rows[i*m->width+j]);
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
    int scalar_mult_success, matrix_matrix_success;
    struct timeval start, stop, overall_t1, overall_t2;
    FILE *fileA, *fileB, *fileR1, *fileR2;
    Matrix *A, *B, *C;
    
    if (argc != 11) {
        printf("Usage: %s <float to multiply> <A matrix height> <A matrix width>"
                                            " <B matrix height> <B matrix width>"
                                            " <A input file> <B input file>"
                                            " <result1 file> <result2 file>\n", argv[0]);
        return 1;
    }
    gettimeofday(&overall_t1, NULL);

    fileA  = openCheck(argv[7], "rb");
    fileB  = openCheck(argv[8], "rb");
    fileR1 = openCheck(argv[9], "wb");
    fileR2 = openCheck(argv[10], "wb");

    set_number_threads(atoi(argv[6]));

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
    printf("Writing first result to %s\n",argv[8]);
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
    printf("Writing second result to %s\n",argv[9]);
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
