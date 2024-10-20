#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N_I 10000 //numero di sotto-intervalli in cui suddividere l'intervallo di integrazione
#define N_THREAD 4 //numero di thread utilizzati

//prototipi funzioni
double funzione_integranda(double);

int main(int argc, char* argv[])
{
	int i; //indici contatori
	double a = 0;
	double b = 2;
	double h = (b - a) / N_I;
	int chunk = (int) N_I / N_THREAD; // numero di sotto-intervalli assegnati a ciascun thread
	double I=0; //variabile che conterrà il valore totale dell'integrale
	double t0, t1, t_tot; // tempo di start, tempo di stop e intervallo di tempo che intercorre tra i due

	if (a < b) {

	//inizio misurazione tempo di esecuzione
	t0 = omp_get_wtime();

#pragma omp parallel for num_threads (N_THREAD) schedule(static,chunk) shared (a,b,h) private(i) reduction(+:I)

		//calcolo parallelo della sommatoria della formula trapezoidale
		for (i = 1; i < N_I; i++) {
			I = I + funzione_integranda(a + i*h);
		}

		//ultima elaborazione della formula trapezoidale
		I = (h / 2)*(funzione_integranda(a) + 2 * I + funzione_integranda(b));

		//fine misurazione tempo di esecuzione
		t1 = omp_get_wtime();
		t_tot = (t1 - t0) * 1000; //tempo di esecuzione in millisecondi
		printf("\n Tempo di calcolo: %f ms \n", t_tot);

		printf("\n Integrale totale: %0.15f  \n",I);

	}
	else {
		printf("\n Errore! L'estremo 'a' deve essere inferiore dell'estremo 'b'!\n");
	}
	
	return 0;
}


//	----- implementazione funzioni -----

// Questa funzione restituisce il valore della funzione assegnata calcolato nel punto x
double funzione_integranda(double x) {
	return 1 / ((x*x*x) - (2 * x) - 5);
	//return -3*(x*x*x*x)+2*(x*x*x)-10*x+1;
	//return x*x;
	//return tan(log(x));
}
