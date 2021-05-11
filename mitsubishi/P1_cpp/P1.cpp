/*
 * An optimization benchmark problem for the energy plant operational planning problem
 * (main file)
 * Written by Takashi Okamoto (takashi@faculty.chiba-u.jp)
 * Last update on June 3rd, 2014
 */
#include "P1.h"

/* Constructor */
P1::P1()
{
	//Set constants
	initialization();

	/* Set common variables */
	N_x = I * (N_t + N_s + 2);
	N_y = I * (N_t + N_s + 2);
	P = 1;
	M = I * (6 + 2 * N_t + 2 * N_s);
	Q = (I - L_t[0]) + (I - L_s[0]) + (I - L_s[1]) + (I - L_g) + (I - L_b) + I;

	//Memory allocation
	x_t = new double *[N_t];
	y_t = new double *[N_t];
	for (int n = 0; n < N_t; n++) {
		x_t[n] = new double [I];
		y_t[n] = new double [I];
	}
	x_s = new double *[N_s];
	y_s = new double *[N_s];
	for (int n = 0; n < N_s; n++) {
		x_s[n] = new double [I];
		y_s[n] = new double [I];
	}
	x_g = new double [I];
	y_g = new double [I];
	x_b = new double [I];
	y_b = new double [I];
	Q_ts = new double [I];
}

/* Destructor */
P1::~P1()
{
	//Release memory
	for (int n = 0; n < N_t; n++) { 
		delete [] x_t[n];
		delete [] y_t[n];
	}
	delete [] x_t;
	delete [] y_t;
	for (int n = 0; n < N_s; n++) {
		delete [] x_s[n];
		delete [] y_s[n];
	}
	delete [] x_s;
	delete [] y_s;
	delete [] x_g;
	delete [] y_g;
	delete [] x_b;
	delete [] y_b;
	delete [] Q_ts;
	delete [] a_t;
	delete [] a_s;
	delete [] b_s;
	delete [] c_s;
	delete [] Q_t_min;
	delete [] Q_t_max;
	delete [] Q_s_min;
	delete [] Q_s_max;
	delete [] L_t;
	delete [] L_s;
	delete [] C_Er;
	delete [] C_Fr;
	delete [] E_L;
	delete [] Q_L;
	delete [] S_L;
	delete [] E_rm;
	delete [] S_rm;
}

/* Evaluation function */
void P1::evaluation(double *x, double *y, double *f, double *g, double *h)
{
	//Substitution of x and y
	int cnt = 0;
	for (int n = 0; n < N_t; n++) {
		for (int i = 0; i < I; i++) {
			x_t[n][i] = x[cnt];
			y_t[n][i] = y[cnt];
			cnt++;
		}
	}
	for (int n = 0; n < N_s; n++) {
		for (int i = 0; i < I; i++) {
			x_s[n][i] = x[cnt];
			y_s[n][i] = y[cnt];
			cnt++;
		}
	}
	for (int i = 0; i < I; i++) {
		x_g[i] = x[cnt];
		y_g[i] = y[cnt];
		cnt++;
	}
	for (int i = 0; i < I; i++) {
		x_b[i] = x[cnt];
		y_b[i] = y[cnt];
		cnt++;
	}

	//Compute Q_ts
	Q_ts[0] = computeQ(0, Q_ts_init);
	for (int i = 1; i < I; i++) {
		Q_ts[i] = computeQ(i, Q_ts[i - 1]);
	}

	//Compute objective function - Eq. (6a)
	f[0] = 0.0;
	for (int i = 0; i < I; i++) {
		f[0] += C_Er[i] * E_r(i) + C_Fr[i] * (x_g[i] + x_b[i]);
	}

	/* Compute constraint conditions */
	//Eq. (6b) and (6c)
	int id_g = 0;
	for (int i = 0; i < I; i++) {
		g[id_g] = Q_ts_min - Q_ts[i];
		id_g++;
	}
	for (int i = 0; i < I; i++) {
		if (i < I - 1) {
			g[id_g] = Q_ts[i] - Q_ts_max1;
		} else {
			g[id_g] = Q_ts[i] - Q_ts_max2;
		}
		id_g++;
	}

	//Eq. (6d)
	int id_h = 0;
	for (int i = 0; i < I; i++) {
		h[id_h] = a_gs * x_g[i] + a_b * x_b[i];
		for (int j = 0; j < N_s; j++) {
			h[id_h] -= f_sj(j, i);
		}
		h[id_h] -= S_L[i];
		h[id_h] -= S_rm[i];
		id_h++;
	}

	//Eq. (6e)
	for (int j = 0; j < N_t; j++) {
		for (int i = 0; i < I; i++) {
			g[id_g] = Q_t_min[j] * y_t[j][i] - x_t[j][i];
			id_g++;
		}
	}
	for (int j = 0; j < N_t; j++) {
		for (int i = 0; i < I; i++) {
			g[id_g] = x_t[j][i] - Q_t_max[j] * y_t[j][i];
			id_g++;
		}
	}

	//Eq. (6f)
	for (int j = 0; j < N_s; j++) {
		for (int i = 0; i < I; i++) {
			g[id_g] = Q_s_min[j] * y_s[j][i] - x_s[j][i];
			id_g++;
		}
	}
	for (int j = 0; j < N_s; j++) {
		for (int i = 0; i < I; i++) {
			g[id_g] = x_s[j][i] - Q_s_max[j] * y_s[j][i];
			id_g++;
		}
	}

	//Eq. (6g)
	for (int i = 0; i < I; i++) {
		g[id_g] = E_g_min * y_g[i] - a_ge * x_g[i];
		id_g++;
	}
	for (int i = 0; i < I; i++) {
		g[id_g] = a_ge * x_g[i] - E_g_max * y_g[i];
		id_g++;
	}

	//Eq. (6h)
	for (int i = 0; i < I; i++) {
		g[id_g] = S_b_min * y_b[i] - a_b * x_b[i];
		id_g++;
	}
	for (int i = 0; i < I; i++) {
		g[id_g] = a_b * x_b[i] - S_b_max * y_b[i];
		id_g++;
	}

	//Eq. (6i)
	for (int j = 0; j < N_t; j++) {
		for (int i = 0; i < I - 1; i++) {
			for (int tau = i + 2; tau < i + L_t[j] + 1; tau++) {
				if (tau < I) {
					h[id_h] = (y_t[j][i + 1] - y_t[j][i]) * (y_t[j][i + 1] - y_t[j][tau]);
					id_h++;
				}
			}
		}
	}

	//Eq. (6j)
	for (int j = 0; j < N_s; j++) {
		for (int i = 0; i < I - 1; i++) {
			for (int tau = i + 2; tau < i + L_s[j] + 1; tau++) {
				if (tau < I) {
					h[id_h] = (y_s[j][i + 1] - y_s[j][i]) * (y_s[j][i + 1] - y_s[j][tau]);
					id_h++;
				}
			}
		}
	}

	//Eq. (6k)
	for (int i = 0; i < I - 1; i++) {
		for (int tau = i + 2; tau < i + L_g + 1; tau++) {
			if (tau < I) {
				h[id_h] = (y_g[i + 1] - y_g[i]) * (y_g[i + 1] - y_g[tau]);
				id_h++;
			}
		}
	}

	//Eq. (6l)
	for (int i = 0; i < I - 1; i++) {
		for (int tau = i + 2; tau < i + L_g + 1; tau++) {
			if (tau < I) {
				h[id_h] = (y_b[i + 1] - y_b[i]) * (y_b[i + 1] - y_b[tau]);
				id_h++;
			}
		}
	}
}

/* Check feasibility */
bool P1::checkFeasibility(double *x, double *y)
{
	/* Check feasibility of y */
	bool feasibility = true;
	for (int n = 0; n < N_y; n++) {
		if (y[n] < 0 || y[n] > 2) {
			feasibility = false;
		}
	}

	/* Compute f, g, and h */
	double *f, *g, *h;
	f = new double[P];
	g = new double[M];
	h = new double[Q];
	evaluation(x, y, f, g, h);

	/* Check inequality conditions */
	for (int m = 0; m < M; m++) {
		if (g[m] > eps) {
			feasibility = false;
		}
	}

	/* Check equality conditions */
	for (int q = 0; q < Q; q++) {
		if (fabs(h[q]) > eps){
			feasibility = false;
		}
	}

	/* Memory release */
	delete [] f;
	delete [] g;
	delete [] h;

	return feasibility;
}

/* Private methods */
void P1::initialization()
{
	//Read constants
	std::ifstream ifs;
	ifs.open("P1_constants.conf");
	std::string tmp;
	ifs >> tmp >> I >> tmp >> N_t >> tmp >> N_s >> tmp >> a_ge >> tmp >> a_gs >> tmp >> a_b >> tmp >> E_g_min >> tmp >> E_g_max >> tmp >> S_b_min >> tmp >> S_b_max >> tmp >> Q_ts_min >> tmp >> Q_ts_max1 >> tmp >> Q_ts_max2 >> tmp >> Q_ts_init >> tmp >> Q_loss >> tmp >> L_g >> tmp >> L_b;
	ifs.close();

	//Read tolerance for constraint violations
	ifs.open("P1_tolerance.conf");
	ifs >> eps;
	ifs.close();

	//Read array constants
	readArray(&a_t, N_t, "P1_a_t.conf");
	readArray(&a_s, N_s, "P1_a_s.conf");
	readArray(&b_s, N_s, "P1_b_s.conf");
	readArray(&c_s, N_s, "P1_c_s.conf");
	readArray(&Q_t_min, N_t, "P1_Q_t_min.conf");
	readArray(&Q_t_max, N_t, "P1_Q_t_max.conf");
	readArray(&Q_s_min, N_s, "P1_Q_s_min.conf");
	readArray(&Q_s_max, N_s, "P1_Q_s_max.conf");
	readArray(&L_t, N_t, "P1_L_t.conf");
	readArray(&L_s, N_s, "P1_L_s.conf");
	readArray(&C_Er, I, "P1_C_Er.conf");
	readArray(&C_Fr, I, "P1_C_Fr.conf");
	readArray(&E_L, I, "P1_E_L.conf");
	readArray(&Q_L, I, "P1_Q_L.conf");
	readArray(&S_L, I, "P1_S_L.conf");
	readArray(&E_rm, I, "P1_E_rm.conf");
	readArray(&S_rm, I, "P1_S_rm.conf");
}
void P1::readArray(double **A, const int N, const std::string &filename)
{
	std::ifstream ifs(filename.c_str());
	*A = new double [N];
	for (int i = 0; i < N; i++) {
		ifs >> (*A)[i];
	}
	ifs.close();
}
void P1::readArray(int **A, const int N, const std::string &filename)
{
	std::ifstream ifs(filename.c_str());
	*A = new int [N];
	for (int i = 0; i < N; i++) {
		ifs >> (*A)[i];
	}
	ifs.close();
}
double P1::E_r(const int i)
{
	double f_t = 0.0;
	for (int j = 0; j < N_t; j++) {
		f_t += a_t[j] * x_t[j][i];
	}
	return f_t + E_L[i] - a_ge * x_g[i] + E_rm[i];
}

double P1::computeQ(const int i, const double Q_ts_i_minus_1)
{
	double Q_ts_i = 0.0;
	for (int j = 0; j < N_t; j++) {
		Q_ts_i -= x_t[j][i];
	}
	for (int j = 0; j < N_s; j++) {
		Q_ts_i -= x_s[j][i];
	}
	Q_ts_i += Q_ts_i_minus_1 + Q_L[i] + Q_loss;
	return Q_ts_i;
}
double P1::f_sj(const int j, const int i)
{
	return x_s[j][i] / (-a_s[j] * x_s[j][i] * x_s[j][i] + b_s[j] * x_s[j][i] + c_s[j]);
}
