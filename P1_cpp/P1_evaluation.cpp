/*
 * P1 class sample
 * Written by Takashi Okamoto (takashi@faculty.chiba-u.jp)
 * Last update on June 3rd, 2014
 */
#include "P1.h"
#include <iostream>
#include <fstream>
using namespace std;

int main()
{
	//preparation
	double *x, *y, *f, *g, *h; //variable declaration
	P1 OF; //create P1 class instance
	x = new double [OF.N_x]; //memory allocation for x
	y = new double [OF.N_y]; //memory allocation for y
	f = new double [OF.P]; //memory allocation for f
	g = new double [OF.M]; //memory allocation for g
	h = new double [OF.Q]; //memory allocation for h

	//set the best known solution
	ifstream ifs("P1_solution_x.txt");
	for (int n = 0; n < OF.N_x; n++) {
		ifs >> x[n];
		if (x[n] < 1.0E-10) {
			y[n] = 0.0;
		} else {
			y[n] = 1.0;
		}
	}

	//evaluation
	OF.evaluation(x, y, f, g, h);

	//output
	cout.precision(10);
	cout << "x = ";
	for (int n = 0; n < OF.N_x; n++) {
		cout << x[n] << " ";
	}
	cout << endl;
	cout << "y = ";
	for (int n = 0; n < OF.N_y; n++) {
		cout << y[n] << " ";
	}
	cout << endl;
	for (int p = 0; p < OF.P; p++) {
		cout << "f" << p + 1 << " = " << f[p] << endl;
	}
	double V = 0.0;
	for (int m = 0; m < OF.M; m++) {
		cout << "g" << m + 1 << " = " << g[m] << endl;
		if (g[m] > 0.0) {
			V += g[m];
		}
	}
	for (int q = 0; q < OF.Q; q++) {
		cout << "h" << q + 1 << " = " << h[q] << endl;
		V += fabs(h[q]);
	}

	//check feasibility
	cout << "Sum of violation = " << V << endl;
	cout << "Tolerance = " << OF.eps << endl;
	if (OF.checkFeasibility(x, y)) {
		cout << "Input solution is feasible.";
	} else {
		cout << "Input solution is infeasible.";
	}

	return 0;
}
