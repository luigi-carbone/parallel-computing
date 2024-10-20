#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#define DIM 100000000 //dimensione dei vettori

//prototipi funzioni
void crea_vettore(double *,int);
void stampa_vettore(double *,int);
double prodotto_scalare(double *,double *,int);

int main(int argc, char *argv[])
{
	//dichiarazioni variabili
	int rank, numprocs;
	int i; //contatore di ciclo
	double p_s = 0; //prodotto scalare totale
	double p_s_partial; //prodotto scalare parziale calcolato da ciascun processore
	double t0, t1,t_tot; // tempo di start, tempo di stop e intervallo di tempo che intercorre tra i due

	//allocazione dinamica dei vettori
	double * x = (double *)malloc(sizeof(double)*DIM); // primo vettore
	double * y = (double *)malloc(sizeof(double)*DIM); // secondo vettore

	//inizializzazione ambiente MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	/*	--- definizione delle dimensioni di ciascun buffer in ricezione --- */
	int * dim_buff = (int *)malloc(sizeof(int)*numprocs); //vettore delle dimensioni di ciascun buffer
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

	//definizione del vettore degli offset in memoria
	int * displ = (int *)malloc(sizeof(int)*numprocs); //vettore dei displacement
	displ[0] = 0;
	for (i = 1; i < numprocs; i++) {
		displ[i] = displ[i - 1] + dim_buff[i - 1];
	}

	//creazione dei vettori di cui effettuare il prodotto scalare
	if (rank == 0) {
		crea_vettore(x, 1);
		crea_vettore(y, 2);
	}

	//allocazione dinamica dei buffer in ricezione
	double *  part_x = (double *)malloc(sizeof(double)*dim_buff[rank]);
	double *  part_y = (double *)malloc(sizeof(double)*dim_buff[rank]);

	//distribuzione da parte del processore con rank 0 delle porzioni di vettore a ciascun processore
	MPI_Scatterv(x, &dim_buff[rank], &displ[rank], MPI_DOUBLE, part_x, dim_buff[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatterv(y, &dim_buff[rank], &displ[rank], MPI_DOUBLE, part_y, dim_buff[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//inizio misurazione del tempo di esecuzione per il calcolo del prodotto scalare
	if (rank == 0) {
		t0 = MPI_Wtime();
	}		

	//ogni processore calcola il proprio prodotto scalare sull'insieme di valori ricevuti
	p_s_partial = prodotto_scalare(part_x,part_y,dim_buff[rank]);

	//il processore con rank 0 raccoglie e somma i prodotti scalari calcolati da tutti i processori 
	MPI_Reduce(&p_s_partial, &p_s, 1, MPI_DOUBLE,MPI_SUM, 0, MPI_COMM_WORLD);

	//fine misurazione del tempo di esecuzione
	if (rank == 0) {
		t1 = MPI_Wtime();
		t_tot = (t1 - t0) * 1000; //tempo di esecuzione in millisecondi
		printf("\n Tempo di calcolo: %f ms \n", t_tot);
	}

	//il processore 0 stampa il prodotto scalare calcolato
	if (rank == 0) {
		printf("\n Sono il processore %d e ho raccolto il seguente prodotto scalare: %0.15f \n",rank,p_s);
	}

	//deallocazione variabili dell'ambiente parallelo
	free(part_x);
	free(part_y);
	free(dim_buff);
	free(displ);
	
	//fine ambiente MPI
	MPI_Finalize();

	//deallocazione variabili dell'ambiente sequenziale
	free(x);
	free(y);

	return 0;
}

//		----- implementazione funzioni -----


// funzione per la creazione di un vettore di dimensione DIM
void crea_vettore(double * a, int seed) {
	srand(seed); // scegliendo un determinato seme si forza la funzione rand a generare una determinata sequenza di valori
	int i; 
	for (i = 0; i <= DIM-1; i++) {
		a[i] = rand() % 10;
		a[i] = a[i] / (1 + (rand() % 9));
	}
}


// funzione per la stampa di un vettore di dimensione nelem
void stampa_vettore(double * a, int nelem) {
	int i; 
	for (i = 0; i < nelem; i++) {
		printf("\n %0.15f ", a[i]);
	}
	printf("\n");
}


// funzione per il calcolo del prodotto scalare tra due vettori
double prodotto_scalare(double * x, double * y, int nelem) {
	int i; 
	double p_s = 0; // variabile che conterrà il prodotto scalare parziale e finale dei due vettori
	for (i = 0; i < nelem; i++) {
		p_s = p_s + x[i] * y[i];
	}
	return p_s;
}

