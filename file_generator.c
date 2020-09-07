/* Utility that generates a binary file of floats to be used with matrix_lib_test.c */
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int i;
    FILE *out;
    float val;
    if (argc != 5) {
        printf("Usage: %s <float value> <Matrix width> <Matrix height> <output file name>\n", argv[0]);
        return 1;
    }
    out = fopen(argv[4], "wb");
    val = atof(argv[1]);
    for (i =0; i<atoi(argv[2])*atoi(argv[3]); i++)
        fwrite(&val, sizeof(float), 1, out);
    fclose(out);
    return 0;
}
