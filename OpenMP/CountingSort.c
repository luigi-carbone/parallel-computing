#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define DIM 100000000 //numero di elementi
#define N_THREAD 2 //numero di thread
#define k 10000  //valore massimo ammissibile

//prototipi funzioni
void crea_vettore(int *);
void stampa_vettore(int *);

int main(int argc, char* argv[])
{
	int i,j; //contatori di ciclo
	double t0, t1, t_tot; // tempo di start, tempo di stop e intervallo di tempo che intercorre tra i due
	int chunk = (int)DIM / N_THREAD;

	//allocazione dinamica dei vettori 
	int * A = (int *)malloc(sizeof(int)*DIM);
	int * B = (int *)malloc(sizeof(int)*DIM); 
	int * C = (int *)malloc(sizeof(int)*(k+1));

	crea_vettore(A);

	//inizio misurazione del tempo di esecuzione per il calcolo del prodotto matrice-vettore
	t0 = omp_get_wtime();

	for (i = 0; i <= k; i++) {
		C[i] = 0;
	}

#pragma omp parallel for num_threads (N_THREAD) shared(C) private (j) schedule (static,chunk) firstprivate(A)
	for (j = 0; j<DIM; j++) {
#pragma omp atomic
		C[A[j]]++;
	}
 
	for (i = 1; i <= k; i++) {
		C[i] = C[i] + C[i - 1];
	}

#pragma omp parallel for num_threads (N_THREAD) shared(A,C) private (j) schedule (static,chunk) 
	for (j = DIM - 1; j >= 0; j--) {
#pragma omp critical
		B[--C[A[j]]] = A[j];
	}
	
	t1 = omp_get_wtime();
	t_tot = (t1 - t0) * 1000; //tempo di esecuzione in millisecondi
	printf("\n Tempo di calcolo: %f ms \n", t_tot);

	//stampa_vettore(B);

	free(A);
	free(B);
	free(C);
	
	return 0;
}

//		----- implementazione funzioni -----


// funzione per la creazione di un vettore di dimensione DIM
void crea_vettore(int * a) {
	srand(1);
	int i;
	for (i = 0; i < DIM; i++) {
		a[i] = rand() % k;
	}
}


// funzione per la stampa di un vettore di dimensione nelem
void stampa_vettore(int * a) {
	int i;
	int q = 0;
	for (i = 0; i < DIM; i++) {
			printf("\n %d ", a[i]);
	}
	printf("\n");
}

