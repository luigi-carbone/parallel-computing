#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define N_I 1000000 //numero di sotto-intervalli in cui suddividere l'intervallo di integrazione

//prototipi funzioni
double funzione_integranda(double);
double formula_trapezoidale(double, double);
double integrale(double, double, int);

int main(int argc, char *argv[])
{
	// dichiarazioni variabili
	int rank, numprocs;
	int i; //contatore di ciclo
	double a = 0; // primo estremo dell'intervallo su cui integrare
	double b = 2; // secondo estremo dell'intervallo su cui integrare
	double integrale_totale = 0; //variabile che conterrà il valore totale dell'integrale
	double t0, t1,t_tot; // tempo di start, tempo di stop e intervallo di tempo che intercorre tra i due

	//inizializzazione ambiente MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); 

	//dichiarazioni variabili ambiente parallelo
	double estremo_intervallo_a, estremo_intervallo_b; //estremi dei sotto-intervalli su cui integrare la funzione
	double I; //variabile che conterrà l'integrale calcolato da ciascun processore
	int sub_N; // numero di sotto-intervalli assegnati a ciascun processore

	/* Il programma deve calcolare l'integrale di una funzione assegnata sull'intervallo [a,b].
	Se a>=b chiaramente il programma non può essere eseguito. */
	if (a >= b) {
		printf("\n L'estremo <a> deve essere minore di <b>! \n ");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	//assegnazione degli estremi di integrazione per tutti i sotto-intervalli
	for (i = 0; i < numprocs; i++) {
		if (i == rank) {
			estremo_intervallo_a = a + (b - a) * i / numprocs;
			estremo_intervallo_b = a + (b - a) * (i + 1) / numprocs;
		}
	}

	/* Definizione degli intervalli di integrazione per ciascun processore.
	Per una ripartizione equa, gni processore deve calcolare l'integrale su un intervallo di ampiezza N_I/numprocs. */
	sub_N = N_I / numprocs;

	//inizio misurazione del tempo di esecuzione per il calcolo dell'integrale
	if (rank == 0) {
		t0 = MPI_Wtime(); //tempo di start
	}

	//ogni processore calcola l'integrale nel sotto-intervallo che gli viene assegnato 
	I = integrale(estremo_intervallo_a, estremo_intervallo_b, sub_N);

	//raccolta e somma, da parte del processore con rank 0, di tutti gli integrali nei sotto-intervalli
	MPI_Reduce(&I, &integrale_totale, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	//fine misurazione del tempo di esecuzione
	if (rank == 0) {
		t1 = MPI_Wtime(); //tempo di stop
		t_tot = (t1 - t0) * 1000; //tempo di esecuzione in millisecondi
		printf("\n Tempo di calcolo: %f ms \n", t_tot);
	}

	//il processore con rank 0 stampa l'integrale totale nell'intervallo di interesse
	if (rank == 0) {
		printf("\n L'integrale totale e': %0.15f \n",integrale_totale);
	}
	
	//fine ambiente MPI
	MPI_Finalize();

	return 0;
}

//		----- implementazione funzioni -----


// Questa funzione restituisce il valore della funzione assegnata calcolato nel punto x
double funzione_integranda(double x) {
	return 1 / ((x*x*x) - (2 * x) - 5);
	//return (x*x)-x+1;
	//return cos(x);
	//return log((x*x+1)/5);
}


/* Questa funzione calcola, con la formula trapezoidale, l'area sottesa al grafico della funzione assegnata 
nell'intervallo [a_i,b_i] fornito in ingresso. */
double formula_trapezoidale(double a_i,double b_i) {
	double area = (b_i - a_i) * (funzione_integranda(b_i) + funzione_integranda(a_i)) / 2;
	return area;
}

/* funzione per la somma di tutti i contributi dati dalle aree calcolate negli N sotto-intervalli 
in cui viene suddiviso l'intervallo di partenza. */
double integrale(double a, double b, int N) {
	int i;
	double somma_integrale = 0; //variabile che conterrà l'integrale parziale e totale della funzione nell'intervallo [a,b]
	for (i = 0; i < N; i++) {
		somma_integrale = somma_integrale + formula_trapezoidale(a + i*(b-a)/N, a + (i+1)*(b-a)/N);
	}
	return somma_integrale;
}

