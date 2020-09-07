CC = gcc
CFLAGS = -Wall

all: file_generator.c matrix_lib.c matrix_lib_test.c timer.c
	$(CC) $(CFLAGS) -o matrix_lib_test matrix_lib.c matrix_lib_test.c timer.c
	$(CC) $(CFLAGS) -o file_generator file_generator.c

debug: file_generator.c matrix_lib.c matrix_lib_test.c timer.c
	$(CC) $(CFLAGS) -g -o matrix_lib_test matrix_lib.c matrix_lib_test.c timer.c
matrix: matrix_lib.c matrix_lib_test.c timer.c
	$(CC) $(CFLAGS) -o matrix_lib_test matrix_lib.c matrix_lib_test.c timer.c
gen: file_generator.c
	$(CC) $(CFLAGS) -o file_generator file_generator.c
