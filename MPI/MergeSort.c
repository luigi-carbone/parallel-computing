#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>
#include <limits.h>

#define DIM 1000 //dimensione del vettore da ordinare

//prototipi
void merge(int *, int, int, int);
void MergeSort(int *, int, int);
void crea_vettore(int *);
void stampa_vettore(int *,int);

int main(int argc, char *argv[])
{
	//dichiarazione variabili
	int rank, numprocs;
	int * x = (int *)malloc(sizeof(int)*DIM); //vettore da ordinare
	int i;
	int n_c; //numero chiamate merge
	double t0, t1, t_tot; // tempo di start, tempo di stop e intervallo di tempo che intercorre tra i due

	//inizializzazione ambiente MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	/* Il programma può essere eseguito solo se il numero di processori impiegato è una potenza di 2 */
	float q_proc =(float) numprocs;
	while (q_proc > 2) {
		q_proc = q_proc / 2;
	}
	if ((q_proc != 2) && (numprocs != 1)) {
		printf("\n Il programma non puo' essere eseguito! Il numero di processori deve essere pari ad una potenza di 2");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	/* fine controllo sul numero di processori*/

	/*	--- definizione dimensione dei buffer in ricezione ---	*/
	int * dim_buff = (int *)malloc(sizeof(int)*numprocs);
	for (i = 0; i < numprocs; i++) {
		dim_buff[i] = DIM / numprocs;
	}

	for (i = 0; i <  (DIM % numprocs); i++) {
		dim_buff[i] = dim_buff[i] +1;
	}
	/*	--- fine definizione buffer ---	*/

	//definizione displacement
	int * displ = (int *)malloc(sizeof(int)*(numprocs+1));
	displ[0] = 0;
	for (i = 1; i <= numprocs; i++) {
		displ[i] = displ[i-1] + dim_buff[i-1];
	}

	//creazione vettore da ordinare
	if (rank == 0) {
		crea_vettore(x);
		/*printf("\n Sono il processore %d . Ho generato il seguente vettore: \n", rank);
		stampa_vettore(x, DIM);*/
	}

	//allocazione del vettore che conterrà la porzione del vettore originale
	int * y = (int *)malloc(dim_buff[rank]*sizeof(int));

	// distribuzione delle porzioni del vettore a tutti i processori del communicator
	MPI_Scatterv(x, &dim_buff[rank], &displ[rank], MPI_INT, y, dim_buff[rank], MPI_INT, 0, MPI_COMM_WORLD);

	//inizio misurazione del tempo di esecuzione per l'operazione di ordinamento
	if (rank == 0) {
		t0 = MPI_Wtime();
	}

	// ciascun processore riordina la porzione ricevuta invocando la funzione Merge Sort
	MergeSort(y,0,dim_buff[rank]-1);

	// il root raccoglie le varie porzioni ordinate
	MPI_Gatherv(y,dim_buff[rank],MPI_INT,x,&dim_buff[rank], &displ[rank],MPI_INT,0,MPI_COMM_WORLD);

	//calcolo del numero di iterazioni di cicli di chiamate alla Merge
	n_c = ceil(log2(numprocs));

	//il processore con rank 0 effettua una serie di Merge su due porzioni ordinate alla volta, costruendo la soluzione finale
	if (rank == 0) {

		int n_p = numprocs / 2; //numero iniziale di iterazioni del ciclo for
		int n = 1; //inizializzazione variabile di proporzione per gli indici 

		while (n_c > 0) {

			int index_merge[3] = { n * 0,n * 1,n * 2 }; //indici della prima invocazione della Merge
			for (i = 0; i < n_p; i++) {

				//fusione di due sequenze ordinate
				merge(x, displ[index_merge[0]], displ[index_merge[1]] - 1, displ[index_merge[2]] - 1);

				//aggiornamento degli indici per la Merge
				index_merge[0] = index_merge[2];
				index_merge[1] = index_merge[0] + n;
				index_merge[2] = index_merge[1] + n;

			}
			//aggiormento variabili per iterazione successiva del ciclo while
			n_p = n_p / 2;
			n_c--;
			n = 2 * n;
		}
	}

	//fine misurazione del tempo di esecuzione
	if (rank == 0) {
		t1 = MPI_Wtime();
		t_tot = (t1 - t0) * 1000; //tempo di esecuzione in millisecondi
		printf("\n Tempo di calcolo: %f ms \n", t_tot);
	}

	//il processore con rank 0 stampa il vettore ordinato
	if (rank == 0) {
		printf("\n Sono il processore %d . Ho ordinato totalmente il vettore:\n", rank);
		stampa_vettore(x, DIM);
	}

	free(dim_buff);
	free(displ);
	free(y);

	//fine ambiente MPI
	MPI_Finalize();

	free(x);

	return 0;
}

/*	----- implementazione funzioni -----	*/

// implementazione della funzione Merge
void merge(int * A, int p, int q, int r) {
	int n1 = q - p + 1;
	int n2 = r - q;
	int i;
	int j;
	int k;

	int * L = (int *)malloc(sizeof(int)*(n1 + 1));
	int * R = (int *)malloc(sizeof(int)*(n2 + 1));

	for (i = 0; i<n1; i++) {
		L[i] = A[p + i];
	}
	for (j = 0; j<n2; j++) {
		R[j] = A[q + j + 1];
	}
	L[n1] = INT_MAX; //limite superiore del tipo int
	R[n2] = INT_MAX;
	i = 0;
	j = 0;
	for (k = p; k <= r; k++) {
		if (L[i] <= R[j]) {
			A[k] = L[i];
			i++;
		}
		else {
			A[k] = R[j];
			j++;
		}
	}
}


// implementazione della funzione Merge Sort
void MergeSort(int * A, int p, int r) {
	int q;
	if (p<r) {
		q = ((p + r) / 2) - ((p + r + 1) % 2);
		MergeSort(A, p, q);
		MergeSort(A, q + 1, r);
		merge(A, p, q, r);
	}
}


// funzione per la creazione random di un vettore di interi di dimensione DIM
void crea_vettore(int * A) {
	srand(1);
	int i;
	for (i = 0; i<DIM; i++) {
		A[i] = rand()%10000;
	}
}


// funzione per la stampa di un vettore di interi di dimensione DIM
void stampa_vettore(int * A,int n) {
	int i;
	for (i = 0; i<n; i++) {
		printf("\n %d", A[i]);
	}
	printf("\n");
}
