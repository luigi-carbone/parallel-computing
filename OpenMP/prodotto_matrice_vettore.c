#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N 15000 //numero di righe
#define M 25000 //numero di colonne
#define N_THREAD 8 //numero di thread utilizzati

//prototipi funzioni
void crea_vettore(double *, int, int);
void stampa_vettore(double *, int);
void crea_matrice(double**, int, int, int);
void stampa_matrice(double**, int, int);

int main(int argc, char* argv[])
{
	int i,j; //contatori di ciclo
	double t0, t1, t_tot; // tempo di start, tempo di stop e intervallo di tempo che intercorre tra i due
	int chunk = (int)N / N_THREAD;

	//allocazione dinamica della matrice
	double * matrix_memory;
	double ** matrice;
	matrice = (double **)malloc(sizeof(double*)*N);
	matrix_memory = (double*)malloc(N*M*(sizeof(double)));
	for (i = 0; i < N; i++) {
		matrice[i] = &(matrix_memory[i*M]);
	}

	//allocazione dinamica dei vettori 
	double * x = (double *)malloc(sizeof(double)*M); //vettore moltiplicatore
	double * y = (double *)malloc(sizeof(double)*N); //vettore risultato

	crea_matrice(matrice, 0, N, M);
	crea_vettore(x, 1, M);
	//stampa_matrice(matrice,N,M);
	//stampa_vettore(x,M);

	//inizio misurazione del tempo di esecuzione per il calcolo del prodotto matrice-vettore
	t0 = omp_get_wtime();

#pragma omp parallel for num_threads (N_THREAD) schedule (static,chunk) default(shared) private (i,j)

		for (i = 0; i < N; i++) {
			y[i] = 0;
			for (j = 0; j < M; j++) {
				y[i] = y[i] + matrice[i][j] * x[j];
			}
		}

	t1 = omp_get_wtime();
	t_tot = (t1 - t0) * 1000; //tempo di esecuzione in millisecondi
	printf("\n Tempo di calcolo: %0.4f ms \n", t_tot);

	/*printf("\n Risultato: \n");
	stampa_vettore(y,N);*/
	
	return 0;
}

//		----- implementazione funzioni -----


// funzione per la creazione di un vettore di dimensione m
void crea_vettore(double * a, int seed, int m) {
	srand(seed); // scegliendo un determinato seme si forza la funzione rand a generare una determinata sequenza di valori
	int i;
	for (i = 0; i < m; i++) {
		a[i] = rand() % 10;
		a[i] = a[i] / (1 + (rand() % 9));
	}
}


// funzione per la stampa di un vettore di dimensione nelem
void stampa_vettore(double * a, int nelem) {
	int i;
	for (i = 0; i < nelem; i++) {
		printf("\n %f ", a[i]);
	}
	printf("\n");
}


// funzione per la creazione di una matrice nxm di elementi random di tipo double
void crea_matrice(double ** a, int seed, int n, int m) {
	srand(seed); // scegliendo un determinato seme si forza la funzione rand a generare una determinata sequenza di valori
	int i, j;
	for (i = 0; i < n; i++) {
		for (j = 0; j < m; j++) {
			a[i][j] = rand() % 10;
			a[i][j] = a[i][j] / (1 + (rand() % 9));
		}
	}
}


//funzione per la stampa di una matrice nxm di elementi double
void stampa_matrice(double ** a, int  n, int m) {
	int i, j;
	for (i = 0; i < n; i++) {
		for (j = 0; j < m; j++) {
			printf(" %0.2f", a[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}
