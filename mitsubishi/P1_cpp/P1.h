/*
 * An optimization benchmark problem for the energy plant operational planning problem
 * (header file)
 * Written by Takashi Okamoto (takashi@faculty.chiba-u.jp)
 * Last update on Dec. 28th, 2013
 * 
 * Variables
 *  x : continuous decision variables
 *  y : discrete decision variables
 *  N_x : number of continuous decision variables
 *  N_y : number of discrete decision variables
 *  P : number of objective funcitons
 *  M : number of inequality conditions
 *  Q : number of equality conditions
 *  eps : tolerance for constraint violations
 *
 * Funcitons
 *  "evaluation" sets objective function values to f, 
 *  inequality condition function values to g, and 
 *  equality condition function values to h.
 *  "checkFeasibility" returns feasibility of decision variables
 */
#include <cmath>
#include <string>
#include <fstream>

class P1 {
public:
	//Constructor
	P1();

	//Public methods
	void evaluation(double *, double *, double *, double *, double *);
	bool checkFeasibility(double *, double *);

	//Constants
	int N_x, N_y, P, M, Q; //common variables
	double eps; //common variables
	int I, N_t, N_s, N, *L_t, *L_s, L_g, L_b;
	double a_ge, a_gs, a_b, *a_t, *a_s, *b_s, *c_s, *Q_t_min, *Q_t_max, *Q_s_min, *Q_s_max, E_g_min, E_g_max, S_b_min, S_b_max, Q_ts_min, Q_ts_max1, Q_ts_max2, Q_ts_init, Q_loss, *C_Er, *C_Fr, *E_L, *Q_L, *S_L, *E_rm, *S_rm;
	
	//Destructor
	~P1();

private:
	//Private methods
	void initialization();
	void readArray(double **, const int, const std::string &);
	void readArray(int **, const int, const std::string &);
	double E_r(const int);
	double computeQ(const int, const double);
	double f_sj(const int, const int);

	//Private members
	double **x_t, **x_s, *x_g, *x_b, **y_t, **y_s, *y_g, *y_b, *Q_ts;
};
