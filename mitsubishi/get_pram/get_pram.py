import configparser 


problem_number = 1
params = configparser.ConfigParser()
params.read('parameters{}.conf'.format(problem_number))

Problemx = 'Problem{}'.format(problem_number)
I = eval(params.get(Problemx, 'I'))
N_t = eval(params.get(Problemx, 'N_t'))
N_s = eval(params.get(Problemx, 'N_s'))
a_ge = eval(params.get(Problemx, 'a_ge'))
a_gs = eval(params.get(Problemx, 'a_gs'))
a_b = eval(params.get(Problemx, 'a_b'))
E_g_min = eval(params.get(Problemx, 'E_g_min'))
E_g_max = eval(params.get(Problemx, 'E_g_max'))
S_b_min = eval(params.get(Problemx, 'S_b_min'))
S_b_max = eval(params.get(Problemx, 'S_b_max'))
Q_ts_min = eval(params.get(Problemx, 'Q_ts_min'))
Q_ts_max1 = eval(params.get(Problemx, 'Q_ts_max1'))
Q_ts_max2 = eval(params.get(Problemx, 'Q_ts_max2'))
Q_ts_init = eval(params.get(Problemx, 'Q_ts_init'))
Q_loss = eval(params.get(Problemx, 'Q_loss'))
L_g = eval(params.get(Problemx, 'L_g'))
L_b = eval(params.get(Problemx, 'L_b'))

a_t = eval(params.get(Problemx, 'a_t'))
a_s = eval(params.get(Problemx, 'a_s'))
b_s = eval(params.get(Problemx, 'b_s'))
c_s = eval(params.get(Problemx, 'c_s'))
Q_t_min = eval(params.get(Problemx, 'Q_t_min'))
Q_t_max = eval(params.get(Problemx, 'Q_t_max'))
Q_s_min = eval(params.get(Problemx, 'Q_s_min'))
Q_s_max = eval(params.get(Problemx, 'Q_s_max'))
L_t = eval(params.get(Problemx, 'L_t'))
L_s = eval(params.get(Problemx, 'L_s'))
C_Er = eval(params.get(Problemx, 'C_Er'))
C_Fr = eval(params.get(Problemx, 'C_Fr'))
E_L = eval(params.get(Problemx, 'E_L'))
Q_L = eval(params.get(Problemx, 'Q_L'))
S_L = eval(params.get(Problemx, 'S_L'))
E_rm = eval(params.get(Problemx, 'E_rm'))
S_rm = eval(params.get(Problemx, 'S_rm'))
eps = eval(params.get(Problemx, 'tolerance'))