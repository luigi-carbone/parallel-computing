#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 15 // numero di righe della matrice e dimensione del vettore risultato
#define M 50 // numero di colonne della matrice e dimensione del vettore moltiplicatore

//prototipi funzioni
void crea_vettore(double *,int, int);
void stampa_vettore(double *,int);
void crea_matrice(double**, int, int, int);
void stampa_matrice(double**, int, int);
double prodotto_riga_vettore(double**, double *, int i);

int main(int argc, char *argv[])
{
	//dichiarazioni variabili
	int rank, numprocs;
	int i; //contatore di ciclo
	double t0, t1,t_tot; // tempo di start, tempo di stop e intervallo di tempo che intercorre tra i due

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

	//inizializzazione ambiente MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	/*	----- definizione delle dimensioni di ciascun buffer in ricezione	----- */
	int * num_r = (int *)malloc(sizeof(int)*numprocs); //vettore numero di righe di ciascun buffer
	int * dim_buff = (int *)malloc(sizeof(int)*numprocs); //vettore delle dimensioni di ciascun buffer

	//ogni buffer deve ricevere N/numprocs righe della matrice per effettuare una ripartizione equa tra i processori
	for (i = 0; i < numprocs; i++) {
		num_r[i] = N / numprocs;
	}

	//se N non è multiplo di numprocs, le righe rimanenti vengono ripartite tra i primi N%numprocs processori
	if (N % numprocs != 0) {
		for (i = 0; i < (N % numprocs); i++) {
			num_r[i] = num_r[i] + 1;
		}
	}

	//ogni buffer in ricezione deve contenere num_r[rank]*M elementi
	for (i = 0; i < numprocs; i++) {
		dim_buff[i] = num_r[i] * M;
	}
	/* ----- fine definizione di ciascun buffer ----- */

	//definizione del vettore dei displacement degli elementi
	int * displ = (int *)malloc(sizeof(int)*numprocs);
	displ[0] = 0;
	for (i = 1; i < numprocs; i++) {
		displ[i] = displ[i - 1] + dim_buff[i - 1];
	}

	//il processore con rank 0 genera la matrice e il vettore da moltiplicare
	if (rank == 0) {
		crea_matrice(matrice, 0, N, M);
		crea_vettore(x, 1, M);
	}

	//allocazione dinamica della matrice in cui ricevere gli elementi
	double * matrix_memory2;
	double ** matrice2;
	matrice2 = (double **)malloc(sizeof(double*)*num_r[rank]);
	matrix_memory2 = (double*)malloc(dim_buff[rank] * (sizeof(double)));
	for (i = 0; i < num_r[rank]; i++) {
		matrice2[i] = &(matrix_memory2[i*M]);
	}

	//distribuzione da parte del processore 0 delle righe a ciascun processore
	MPI_Scatterv(&matrice[0][0], &dim_buff[rank], &displ[rank], MPI_DOUBLE, &matrice2[0][0], dim_buff[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//invio in broadcast del vettore a tutti i processori del communicator
	MPI_Bcast(x, M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//inizio misurazione del tempo di esecuzione per il calcolo del prodotto matrice-vettore
	if (rank == 0) {
		t0 = MPI_Wtime();
	}

	//ogni processore effettua la moltiplicazione della sotto-matrice ricevuta con il vettore dato
	double * part_y = (double *)malloc(sizeof(double)*num_r[rank]); //sotto-vettore del risultato finale
	for (i = 0; i < num_r[rank]; i++) {
		part_y[i] = prodotto_riga_vettore(matrice2,x,i);
	}

	//calcolo dei displacement all'interno del vettore risultato
	int * p = (int *)malloc(sizeof(int)*numprocs);
	p[0] = 0; 
	for (i = 1; i < numprocs; i++) {
		p[i] = p[i - 1] + num_r[i-1];
	}

	//tutti i processori inviano al processore con rank 0 i sotto-vettori calcolati
	MPI_Gatherv(part_y, num_r[rank], MPI_DOUBLE, &y[rank], &num_r[rank], &p[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//fine misurazione del tempo di esecuzione
	if (rank == 0) {
		t1 = MPI_Wtime();
		t_tot = (t1 - t0) * 1000; //tempo di esecuzione in millisecondi
		printf("\n Tempo di calcolo: %f ms \n", t_tot);
	}
	
	//il processore 0 stampa il vettore risultato del prodotto matrice-vettore
	if (rank == 0) {
		printf("\n Sono il processore con rank %d . Ho calcolato il seguente prodotto \n",rank);
		stampa_vettore(y, N);
	}
	
	//deallocazione variabili dell'ambiente parallelo
	free(num_r);
	free(dim_buff);
	free(displ);
	free(matrice2);
	free(matrix_memory2);
	free(part_y);
	free(p);
	
	//fine ambiente MPI
	MPI_Finalize();

	//deallocazione variabili dell'ambiente sequenziale
	free(matrice);
	free(matrix_memory);
	free(x);
	free(y);

	printf("\n\n");
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


//funzione per il calcolo del prodotto tra la riga i-esima di una matrice m e il vettore v
double prodotto_riga_vettore(double** m, double * v, int i) {
	int j;
	double elem = 0;
	for (j = 0; j < M; j++) {
		elem = elem + m[i][j] * v[j];
	}
	return elem;
}

