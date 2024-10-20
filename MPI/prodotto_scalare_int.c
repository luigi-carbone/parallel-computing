#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#define DIM 100000 //dimensione dei vettori

//prototipi funzioni
void crea_vettore(int *,int);
void stampa_vettore(int *,int);
int prodotto_scalare(int *,int *,int);

int main(int argc, char *argv[])
{
	//dichiarazioni variabili
	int rank, numprocs;
	int i; //contatore di ciclo
	int p_s = 0; //prodotto scalare totale
	int p_s_partial; //prodotto scalare parziale calcolato da ciascun processore
	double t0, t1,t_tot;

	//allocazione dinamica dei vettori
	int * x = (int*)malloc(sizeof(int)*DIM); // primo vettore
	int * y = (int*)malloc(sizeof(int)*DIM); // secondo vettore

	//inizializzazione ambiente MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	/*---definizione delle dimensioni di ciascun buffer in ricezione---*/
	int * dim_buff = (int *)malloc(sizeof(int)*numprocs); // vettore delle dimensioni di ciascun buffer
	for (i = 0; i < numprocs; i++) {
		//ogni buffer deve contenere DIM/numprocs elementi per effettuare una ripartizione equa tra i processori
		dim_buff[i] = DIM / numprocs;
	}

	//se DIM non è multiplo del numero di processori, gli elementi rimanenti vengono ripartiti ad alcuni processori
	if (DIM % numprocs != 0) {
		for (i = 0; i < (DIM % numprocs); i++) {
			dim_buff[i] = dim_buff[i] + 1;
		}
	}
	/* fine definizione di ciascun buffer */

	//definizione del vettore degli offset in memoria degli elementi
	int * displ = (int *)malloc(sizeof(int)*numprocs); // vettore displacement degli elmenti in memoria
	displ[0] = 0;
	for (i = 1; i < numprocs; i++) {
		displ[i] = displ[i - 1] + dim_buff[i - 1];
	}

	//creazione dei vettori di cui effettuare il prodotto scalare
	if (rank == 0) {
		crea_vettore(x, 0);
		crea_vettore(y, 10);
	}

	//allocazione dinamica dei vettori buffer in ricezione
	int *  part_x = (int*)malloc(sizeof(int)*dim_buff[rank]);
	int *  part_y = (int*)malloc(sizeof(int)*dim_buff[rank]);

	//invio da parte del processore con rank 0 degli elementi a ciascun processore
	MPI_Scatterv(x, &dim_buff[rank], &displ[rank], MPI_INT, part_x, dim_buff[rank], MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatterv(y, &dim_buff[rank], &displ[rank], MPI_INT, part_y, dim_buff[rank], MPI_INT, 0, MPI_COMM_WORLD);

	//inizio misurazione del tempo di esecuzione per il calcolo del prodotto scalare
	if (rank == 0) {
		t0 = MPI_Wtime();
	}
		
	//ogni processore calcola il proprio prodotto scalare sull'insieme di valori ricevuti
	p_s_partial = prodotto_scalare(part_x,part_y,dim_buff[rank]);

	//il processore con rank 0 raccoglie e somma i prodotti scalari calcolati da tutti i processori 
	MPI_Reduce(&p_s_partial, &p_s, 1, MPI_INT,MPI_SUM, 0, MPI_COMM_WORLD);

	//fine misurazione del tempo di esecuzione
	if (rank == 0) {
		t1 = MPI_Wtime();
		t_tot = t_tot + (t1 - t0) * 1000; //tempo in ms
		printf("\n Tempo di calcolo: %f ms \n", t_tot);
	}

	//il processore 0 stampa il prodotto scalare calcolato
	if (rank == 0) {
		printf("\n Sono il processore %d e ho raccolto il seguente prodotto scalare: %d \n",rank,p_s);
	}
	
	//deallocazione variabili ambiente sequenziale
	free(part_x);
	free(part_y);
	
	//fine ambiente MPI
	MPI_Finalize();

	//deallocazione variabili ambiente sequenziale
	free(x);
	free(y);
	free(dim_buff);
	free(displ);

	return 0;
}

//		----- implementazione funzioni -----

// funzione per la creazione di un vettore di dimensione DIM
void crea_vettore(int * a,int seed) {
	srand(seed); // scegliendo un determinato seme si forza la funzione rand a generare una determinata sequenza di valori
	int i; 
	for (i = 0; i <= DIM-1; i++) {
		a[i] = rand()%10;
	}
}


// funzione per la stampa di un vettore di dimensione nelem
void stampa_vettore(int * a,int nelem) {
	int i; 
	for (i = 0; i < nelem; i++) {
		printf("\n %d ", a[i]);
	}
	printf("\n");
}


// funzione per il calcolo del prodotto scalare tra due vettori
int prodotto_scalare(int * x, int * y,int nelem) {
	int i; 
	int p_s = 0; // variabile che conterrà il prodotto scalare parziale e finale dei due vettori
	for (i = 0; i < nelem; i++) {
		p_s = p_s + x[i] * y[i];
	}
	return p_s;
}

