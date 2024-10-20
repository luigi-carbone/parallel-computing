#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define dim 2 //numero di dimensioni della griglia
#define N  5000//numero di righe costituenti la matrice
#define M 5000 //numero di colonne costituenti la matrice

//prototipi funzioni
void crea_vettore(double *, int, int);
void stampa_vettore(double *, int);
void crea_matrice(double**, int, int, int);
void stampa_matrice(double**, int,int);
void stampa_vettore_m(double *, int, int);
double prodotto_riga_vettore(double*, double *, int, int, int);

int main(int argc, char *argv[])
{
	//dichiarazione variabili
	int rank, numprocs;
	MPI_Datatype column;
	MPI_Datatype new_column;
	int i,j; //contatori di ciclo
	double t0, t1, t_tot; // tempo di start, tempo di stop e intervallo di tempo che intercorre tra i due

	//allocazione dinamica della matrice
	double * matrix_memory;
	double ** matrice;
	matrice = (double **)malloc(sizeof(double*)*N);
	matrix_memory = (double*)malloc(N*M*(sizeof(double)));
	for (i = 0; i < N; i++) {
		matrice[i] = &(matrix_memory[i*M]);
	}

	//allocazione dinamica del vettore
	double * x = (double*)malloc(M * sizeof(double)); //vettore moltiplicatore
	double * y = (double*)malloc(N * sizeof(double)); //vettore risultato

	//inizializzazione ambiente MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//parametri delle topologia cartesiana
	int q = (int)sqrt(numprocs); //numero di processori lungo un lato della griglia
	int ndim[dim] = { q,q };
	int period[dim] = { 1,1 }; //periodicità della griglia
	MPI_Comm grid;
	MPI_Comm riga;
	MPI_Comm colonna;
	int coords[dim]; // coordinate dei processori all'interno della griglia
	int rdims[dim]; //varying coords

	//se numprocs non è un quadrato perfetto il programma non può funzionare
	if (q*q != numprocs) {
		printf("\n Errore! Il numero di processori deve essere un quadrato perfetto");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	//creazione topologia cartesiana (griglia di processori qxq)
	MPI_Cart_create(MPI_COMM_WORLD, dim, ndim, period, 0, &grid);

	//restituzione delle coordinate a ciascun processore della griglia
	MPI_Cart_coords(grid, rank, dim, coords);

	//Creazione della sottotopologia cartesiana riga
	rdims[0] = 0;
	rdims[1] = 1;
	MPI_Cart_sub(grid, rdims, &riga);

	//Creazione della sottopologia cartesiana colonna
	rdims[0] = 1;
	rdims[1] = 0;
	MPI_Cart_sub(grid, rdims, &colonna);

	/*	--- definizione delle dimensioni di ciascun buffer in ricezione per la ditribuzione delle righe	--- */
	int * num_righe_blocco = (int *)malloc(sizeof(int)*q); //vettore che conterrà il numero di righe di ciascun blocco
	for (i = 0; i < q; i++) {
		//ogni buffer deve contenere N/q righe per effettuare una ripartizione equa tra i processori del communicator
		num_righe_blocco[i] = N / q;
	}

	//se N non è multiplo di q, le righe rimanenti vengono ripartite tra alcuni processori
	if (N % q != 0) {
		for (i = 0; i < (N % q); i++) {
			num_righe_blocco[i] = num_righe_blocco[i] + 1;
		}
	}

	int * dim_buff_r = (int *)malloc(sizeof(int)*q); // vettore delle dimensioni di ciascun buffer di righe
	for (i = 0; i < q; i++) {
		//ogni buffer dovrà contenere un numero di elementi pari al numero di righe assegnate al processore x il numero di colonne totale della matrice
		dim_buff_r[i] = num_righe_blocco[i] * M; 
	}
	/* --- fine definizione di ciascun buffer riga --- */

	//definizione del vettore dei displacement degli elementi di ciascun blocco di righe
	int * displ_r = (int *)malloc(sizeof(int)*q);
	displ_r[0] = 0;
	for (i = 1; i < q; i++) {
		displ_r[i] = displ_r[i - 1] + dim_buff_r[i - 1];
	}

	//allocazione dinamica della matrice in cui ricevere i blocchi di righe
	double * matrix_memory2;
	double ** matrice_righe;
	matrice_righe = (double **)malloc(sizeof(double*)*num_righe_blocco[coords[0]]);
	matrix_memory2 = (double*)malloc(dim_buff_r[coords[0]] * sizeof(double));
	for (i = 0; i < num_righe_blocco[coords[0]]; i++) {
		matrice_righe[i] = &(matrix_memory2[i*M]);
	}

	//il processore 0 crea la matrice da distribuire
	if (rank == 0) {
		crea_matrice(matrice, 0, N, M);
		crea_vettore(x, 1, M);
	}
	
	if (coords[1] == 0) {
		//distribuzione dei blocchi di righe ai processori della prima colonna della griglia
		MPI_Scatterv(&matrice[0][0], &dim_buff_r[coords[0]], &displ_r[coords[0]], MPI_DOUBLE, &matrice_righe[0][0], dim_buff_r[coords[0]], MPI_DOUBLE, 0, colonna);
		
		//distribuzione del vettore x (moltiplicatore) a tutti i processori della prima colonna della griglia
		MPI_Bcast(x, M, MPI_DOUBLE, 0, colonna);
	}

	
	/*-- - definizione delle dimensioni di ciascun buffer di colonne in ricezione --- */
	int * num_colonne_blocco = (int *)malloc(sizeof(int)*numprocs); // vettore che conterrà il numero di colonne di ciascun blocco
	for (i = 0; i < q; i++) {
		//ogni buffer deve contenere M/q colonne per effettuare una ripartizione equa
		num_colonne_blocco[i] = M / q;
	}

	//se M non è multiplo di q, le colonne rimanenti vengono ripartite ad alcuni processori
	if (M % q != 0) {
		for (i = 0; i < (M % q); i++) {
			num_colonne_blocco[i] = num_colonne_blocco[i] + 1;
		}
	}


	int * dim_buff_c = (int *)malloc(sizeof(int)*numprocs); //vettore delle dimensioni di ciascun blocco di colonne
	for (i = 0; i < numprocs; i++) {
		dim_buff_c[i] = num_righe_blocco[i / q] * num_colonne_blocco[i%q];
	}
	/* fine definizione di ciascun buffer colonna */

	//definizione nuovo datatype (colonna del blocco)
	MPI_Type_vector(num_righe_blocco[coords[0]], 1, M, MPI_DOUBLE, &column);
	MPI_Type_commit(&column); // inizializzazione del datatype
	MPI_Type_create_resized(column, 0, sizeof(double), &new_column); // si ridefinisce l'estensione del precedente tipo
	MPI_Type_commit(&new_column); // inizializzazione del datatype

	//definizione del vettore dei displacement delle colonne di ciascun blocco
	int * displ_c = (int *)malloc(sizeof(int)*q);
	displ_c[0] = 0;
	for (i = 1; i < q; i++) {
		displ_c[i] = displ_c[i - 1] + num_colonne_blocco[i - 1];
	}

	double * blocco = (double*)malloc(sizeof(double)*dim_buff_c[rank]); //vettore che conterrà gli elementi di ciascun blocco
	double * part_x = (double*)malloc(sizeof(double) * num_colonne_blocco[coords[1]]); //vettore che conterrà una porzione di x

	//distribuzione dei blocchi a ciascun processore della griglia
	MPI_Scatterv(&matrice_righe[0][0], num_colonne_blocco, displ_c, new_column, blocco, dim_buff_c[rank], MPI_DOUBLE, 0, riga);

	//distribuzione delle porzioni di x a ciascun processore della griglia
	MPI_Scatterv(x, &num_colonne_blocco[coords[1]], &displ_c[coords[1]], MPI_DOUBLE, part_x, num_colonne_blocco[coords[1]], MPI_DOUBLE, 0, riga);

	//sincronizzazione dei processi
	MPI_Barrier(MPI_COMM_WORLD);

	double * r = (double *)malloc(num_righe_blocco[coords[0]] * sizeof(double)); //vettore che conterrà il prodotto di ciascun blocco per la porzione di vettore ricevuta dal processore
	double * part_y = (double *)malloc(num_righe_blocco[coords[0]] * sizeof(double)); //vettore in cui viene raccolta la somma dei vettori "r" di ciascun processore relativi ad una riga della griglia

	//inizio misurazione del tempo di esecuzione per il calcolo del prodotto matrice-vettore
	if (rank == 0) {
		t0 = MPI_Wtime();
	}

	//ogni processore calcola il proprio contributo per il prodotto finale
	for (i = 0; i < num_righe_blocco[coords[0]]; i++) {
		r[i] = prodotto_riga_vettore(blocco, part_x, i, num_colonne_blocco[coords[1]], num_righe_blocco[coords[0]]);
	}

	//il processore 0 effettua la somma dei vettori calcolati da ciascun processore lungo una riga della griglia
	MPI_Reduce(r, part_y, num_righe_blocco[coords[0]], MPI_DOUBLE, MPI_SUM, 0, riga);	

	//definizione dei displacement all'interno del vettore risultato
	int * displ = (int *)malloc(num_righe_blocco[coords[0]] * sizeof(int));
	displ[0] = 0;
	for (i = 1; i < num_righe_blocco[coords[0]]; i++) {
		displ[i] = displ[i - 1] + num_righe_blocco[i - 1];
	}

	//tutti i processori inviano al processore con rank 0 i sotto-vettori calcolati
	MPI_Gatherv(part_y, num_righe_blocco[coords[0]], MPI_DOUBLE, y, &num_righe_blocco[coords[0]], &displ[coords[0]], MPI_DOUBLE, 0,colonna);

	//fine misurazione del tempo di esecuzione
	if (rank == 0) {
		t1 = MPI_Wtime();
		t_tot = (t1 - t0) * 1000; // tempo di esecuzione in millisecondi
		printf("\n Tempo di calcolo: %f ms \n", t_tot);
	}

	//il processore 0 stampa il vettore risultato
	/*if (rank == 0) {
		printf("\n Sono il processore %d e ho raccolto e calcolato il seguente prodotto: ",rank);
		stampa_vettore(y, N);
	}*/

	//deallocazione variabili dell'ambiente parallelo*/
	free(matrice_righe);
	free(matrix_memory2);
	free(dim_buff_r);
	free(dim_buff_c);
	free(num_righe_blocco);
	free(num_colonne_blocco);
	free(displ_r);
	free(displ_c);
	free(displ);
	free(blocco);

	//liberazione datatype
	MPI_Type_free(&column);
	MPI_Type_free(&new_column);

	//fine ambiente MPI
	MPI_Finalize();

	//deallocazione variabili dell'ambiente sequenziale
	free(matrice);
	free(matrix_memory);

	return 0;
}

//	----- implementazione funzioni -----


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
		printf("\n %0.2f ", a[i]);
	}
	printf("\n");
}


//funzione per la creazione di una matrice n x m di elementi random
void crea_matrice(double ** a, int seed, int n, int m) {
	srand(seed);
	int i, j;
	for (i = 0; i < n; i++) {
		for (j = 0; j < m; j++) {
			a[i][j] = rand() % 10;
			a[i][j] = a[i][j] / (1 + (rand() % 9));
		}
	}
}


//funzione per la stampa di una matrice n x m
void stampa_matrice(double ** a, int  n, int m) {
	int i, j;
	for (i = 0; i < n; i++) {
		for (j = 0; j < m; j++) {
			printf(" %0.2f", a[i][j]);
		}
		printf("\n");
	}
}


//funzione per la stampa di un vettore di double di cardinalità n*m in forma matriciale
void stampa_vettore_m(double * a, int n, int m) {
	int i, j;
	for (i = 0; i < n; i++) {
		for (j = 0; j < m; j++) {
			printf(" %0.2f", a[i + j*n]);
		}
		printf("\n");
	}
	printf("\n");
}


//funzione per il calcolo del prodotto tra la riga j-esima di una matrice v (sottoforma di vettore) e il vettore x
double prodotto_riga_vettore(double * v, double * x, int j, int m,int n) {
	int i;
	double elem = 0;
	for (i = 0; i < m; i++) {
		elem = elem + x[i] * v[j + i*n];
	}
	return elem;
}

