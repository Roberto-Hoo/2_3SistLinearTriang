/**/

#include <omp.h>
#include <cstdio>
#include <iostream>
#include <ctime>
#include <algorithm>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_spmatrix.h>

using namespace std;

#define NumThreads 8
int np, pid;
int ini, fim;
char caracter;
int n = NumThreads * 2;

/*
int M1[NumThreads];
int M2[NumThreads];
int M3[NumThreads];
int M4[NumThreads];
*/

// vetores
gsl_spmatrix *A = gsl_spmatrix_alloc(n, n);
gsl_vector *b = gsl_vector_alloc(n);
gsl_vector *x = gsl_vector_alloc(n);


#pragma omp threadprivate(np, pid, ini, fim)

int main() {



    // inicialização
    printf("Inicializando ... \n");

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            gsl_spmatrix_set(A, i, j, pow(-1.0, i + j) * (i + j) / (i * j + 1));
        }
        gsl_spmatrix_set(A, i, i, (pow(i - n / 2, 2) + 1) * 2 / n);
        gsl_vector_set(b, i, pow(-1.0, i) / (i + 1));
    }

    // Print the values of A using GSL print functions
    cout << "A = \n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            //printf("A(%d,%d) = %g\n", i, j, gsl_matrix_get(A, i, j));
            printf("%5.1f", gsl_spmatrix_get(A, i, j));
        }
        printf("\n");
    }

    // Print the values of b using GSL print functions
    cout << "\nb-Transposta = \n";
    for (int i = 0; i < n; i++) {
        printf("%5.1f", gsl_vector_get(b, i));
    }


    printf("\nfeito.\n");
    printf("Executando em paralelo ... \n");

    time_t t = time(nullptr);

    omp_set_num_threads(NumThreads);
#pragma omp parallel  //#pragma omp threadprivate(np,pid, ini, fim)
    {
        np = omp_get_num_threads();
        pid = omp_get_thread_num();
        ini = pid * n / np;
        fim = (pid + 1) * n / np;
        if (pid == np - 1)
            fim = n;
        printf("\n np=%2d    pid=%2d     ini=%2d    fim=%2d", np, pid, ini, fim);
    }

    for (int i = 0; i < n; i++) {
        double s = 0;
#pragma omp parallel reduction(+: s)
        {
            for (int j = max(0, ini); j < i and j < fim; j++)
                s += gsl_spmatrix_get(A, i, j) * gsl_vector_get(x, j);
        }
        gsl_vector_set(x, i, (gsl_vector_get(b, i) - s) / gsl_spmatrix_get(A, i, i));
    }

    // Print the values of b using GSL print functions
    cout << "\nx-Transposta = \n";
    for (int i = 0; i < n; i++) {
        printf("%5.1f", gsl_vector_get(x, i));
    }

    t = time(nullptr) - t;

    printf("\n Tempo de computacao = %ld s\n", t);


    gsl_spmatrix_free(A);
    gsl_vector_free(b);
    gsl_vector_free(x);


    cout << "\n\n Tecle uma tecla e apos Enter para finalizar...\n";
    cin >> caracter;

    return 0;
}