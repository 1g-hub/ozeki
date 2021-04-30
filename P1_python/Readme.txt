## C++ source codes of the optimization benchmark problem for the energy plant operational planning problem ##
These codes are edited by the investigating R&D commitee on new development of computational intelligence techniques and their applications to industrial systems, the institute of electrical engineers of Japan.

These codes are based on the energy plant operational planning problem in [1].
[1] T. Okamoto, N. Adachi, R. Suzuki, S. Koakutsu, and H. Hirata: "The energy plant operational planning problem and applications of optimization methods to the problem", Proc. of the 2014 Annual Meeting on the Institute of Electrical Engineers of Japan (Mar. 2014) [in Japanese]

-- Files --
# P1.cpp: P1 (the energy plant operational planning problem) class main file.
# P1.h: P1 class header file.
# P1_evaluation.cpp: sample file to test P1 class.
# P1_x.txt: the solution evaluated in P1_evaluation.cpp.
# P1_tolerance.conf: tolerance for constraint violations.
# P1_zzz.conf: constants read in the private method "P1::initialization" of P1.cpp.

-- Implementing the sample code --
# Visual C++
	cl P1.cpp P1_evaluation.cpp /FeP1.exe /EHsc
	P1.exe
# gcc
	g++ P1.cpp P1_evaluation.cpp -o P1.out
	./P1.out

-- Public methods --
# void P1::evaluation(double *x, double *y, double *f, double *g, double *h)
  This function sets objective function values f(x, y) to f, inequality condition function values g(x, y) to g, and equality condition function values h(x, y) to h.
# bool P1::checkFeasibility(double *x, double *y)
  This function returns feasibility of decision variables.

-- Public properties --
# int P1.N_x : number of continuous decision variables
# int P1.N_y : number of discrete decision variables
# int P1.P : number of objective funcitons 目的関数の数
# int P1.M : number of inequality conditions
# int P1.Q : number of equality conditions
# double P1.eps : tolerance for constraint violations
