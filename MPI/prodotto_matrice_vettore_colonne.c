#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 15000 // numero di righe della matrice e dimensione del vettore risultato
#define M 25000 // numero di colonne della matrice e dimensione del vettore moltiplicatore

//prototipi funzioni
void crea_vettore(double *,int, int);
void stampa_vettore(double *,int);
void crea_matrice(double**, int, int, int);
void stampa_matrice(double**, int, int);
double prodotto_riga_vettore(double*, double *, int, int);

int main(int argc, char *argv[])
{
	//dichiarazioni variabili
	int rank, numprocs;
	int i; //contatore di ciclo
	double t0, t1,t_tot; // tempo di start, tempo di stop e intervallo di tempo che intercorre tra i due
	MPI_Datatype colonna;
	MPI_Datatype new_colonna;

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

	/* --- definizione delle dimensioni di ciascun buffer in ricezione --- */
	int * dim_buff = (int *)malloc(sizeof(int)*numprocs); //alloca memoria per vettore delle dimensioni di ciascun buffer
	for (i = 0; i < numprocs; i++) {
		//ogni buffer deve contenere M/numprocs colonne per effettuare una ripartizione equa tra i processori
		dim_buff[i] = M / numprocs;
	}

	//se M non è multiplo di numprocs, le colonne rimanenti vengono ripartite tra i primi M%numprocs processori
	if (M % numprocs != 0) {
		for (i = 0; i < (M % numprocs); i++) {
			dim_buff[i] = dim_buff[i] + 1;
		}
	}
	/*	--- fine definizione di ciascun buffer---	*/

	//definizione del vettore dei displacement in memoria degli elementi
	int * displ = (int *)malloc(sizeof(int)*numprocs);
	displ[0] = 0;
	for (i = 1; i < numprocs; i++) {
		displ[i] = displ[i - 1] + dim_buff[i - 1];
	}

	//il processore con rank 0 genera la matrice e il vettore da moltiplicare
	if (rank == 0) {
		crea_matrice(matrice, 0, N, M);
		//printf("\n Sono il processore con rank %d . Ho generato la seguente matrice: \n",rank);
		//stampa_matrice(matrice,N,M);
		crea_vettore(x, 1, M);
		//printf("\n Sono il processore con rank %d . Ho generato il seguente vettore: \n",rank);
		//stampa_vettore(x,M);
	}

	//definizione nuovo datatype
	MPI_Type_vector(N, 1, M, MPI_DOUBLE, &colonna);
	MPI_Type_commit(&colonna); // inizializzazione del datatype
	MPI_Type_create_resized(colonna, 0, sizeof(double), &new_colonna); // si ridefinisce l'estensione del precedente tipo
	MPI_Type_commit(&new_colonna); // inizializzazione del datatype

	double * vett = (double*)malloc(N * sizeof(double)*dim_buff[rank]); // vettore per ricezione matrice distribuita a colonne
	double * part_x = (double*)malloc(sizeof(double)*dim_buff[rank]); // vettore per ricezione parti di x (cioè del vettore moltiplicatore)
	double * r = (double*)malloc(sizeof(double)*N); // vettore per calcolo prodotto parziale del risultato di ogni processore

	//il processore con rank 0 distribuisce le colonne ai processori
	MPI_Scatterv(&matrice[0][0], &dim_buff[rank], &displ[rank], new_colonna, vett, N*dim_buff[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//il processore con rank 0 distribuisce le porzioni di vettore ai processori
	MPI_Scatterv(&x[0], &dim_buff[rank], &displ[rank], MPI_DOUBLE, part_x, dim_buff[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//inizio misurazione del tempo di esecuzione per il calcolo del prodotto matrice-vettore
	if (rank == 0) {
		t0 = MPI_Wtime();
	}

	//ogni processore calcola il proprio contributo per il prodotto finale
		for (i = 0; i < N; i++) {
			r[i] = prodotto_riga_vettore(vett,part_x,i,dim_buff[rank]);
		}

	//il processore 0 effettua la somma dei vettori calcolati da ciascun processore
	MPI_Reduce(r, y, N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	//fine misurazione del tempo di esecuzione
	if (rank == 0) {
		t1 = MPI_Wtime();
		t_tot = (t1 - t0) * 1000; // tempo di esecuzione in millisecondi
		printf("\n Tempo di calcolo: %f ms \n", t_tot);
	}

	//il processore 0 stampa il vettore risultato
	/*if (rank == 0) {
		printf("\n Risultato: \n");
		stampa_vettore(y, N);
	}*/

	//deallocazione variabili dell'ambiente parallelo
	free(dim_buff);
	free(displ);
	free(vett);
	free(part_x);
	free(r);
	
	//fine ambiente MPI
	MPI_Finalize();

	//deallocazione variabili dell'ambiente sequenziale
	free(matrice);
	free(matrix_memory);
	free(x);
	free(y);

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
			printf(" %0.4f", a[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}


//funzione per il calcolo del prodotto tra la riga j-esima di una matrice v (sottoforma di vettore) e il vettore x
double prodotto_riga_vettore(double * v, double * x, int j, int m) {
	int i;
	double elem=0;
		for (i = 0; i < m; i++) {
			elem = elem + x[i] * v[j + i*N];
		}
	return elem;
}
