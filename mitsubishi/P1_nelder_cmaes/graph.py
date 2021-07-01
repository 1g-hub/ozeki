import numpy as np
import matplotlib.pyplot as plt
import P1

 
# 折れ線グラフを出力
x = [1.5000000000057772, 1.5000000000048574, 1.5000000000023128, 1.5000000000101161, 1.5000000000018952, 1.4999999999999944, 1.5000000000079845, 1.5000000000046145, 1.5000000000128668, 1.500000000001935, 1.5000000000030138, 1.500000000000841, 1.5000000000061222, 1.5000000000051144, 1.5000000000005838, 1.5000000000084475, 1.5000000000092082, 1.5000000000083702, 1.5000000000022053, 1.500000000004406, 1.5000000000041342, 1.5000000000023974, 1.5000000000115445, 1.5000000000051692, 4.500000000006956, 4.500000000006776, 4.50000000000537, 4.500000000002049, 4.499999999999645, 4.50000000000171, 4.500000000008152, 4.500000000002143, 14.999999999996637, 13.317660750982686, 5.720236521105866, 4.50000000000564, 4.5000000000072164, 4.500000000002209, 4.500000000001247, 4.5000000000092175, 4.4999999999996305, 4.500000000003034, 4.500000000002537, 4.500000000001466, 4.500000000005976, 14.679294288909906, 4.500000000006334, 4.500000000002674, 4.500000000005819, 4.500000000001366, 4.5000000000021805, 4.500000000001621, 4.500000000001713, 4.500000000007383, 4.500000000000414, 4.500000000002311, 14.999999999997327, 14.495437517370405, 5.085901185685391, 4.500000000000594, 4.49999999999986, 4.500000000008649, 4.50000000000044, 4.500000000004504, 4.500000000005505, 4.5000000000047855, 4.500000000000763, 4.50000000000215, 4.499999999999597, 14.638556176635346, 4.500000000001802, 4.500000000005412, -3.854501263269444e-11, 1.5208065300794813e-11, -2.9778755940426296e-11, -2.956727202889841e-11, 4.490097368953275e-14, -1.141019450734075e-11, 6.654679913667691e-12, -2.044441743145296e-11, 3039.4152632350715, 3678.9322267100706, 3678.9322267104412, 3678.9322267107964, 3678.9322267105035, 3678.9322267113657, 3678.9322267106877, 3678.932226710886, 3678.9322267111784, 3678.9322267114967, 3678.9322267109487, 3678.9322267107586, 3678.9322267108587, 3678.93222671084, -3.444652046750482e-11, -4.0659910821239134e-11, 150.27905365452403, 150.27905365454217, 150.27905365444366, 150.2790536543838, 150.27905365451514, 150.27905365452668, 200.44711666555153, 250.61517967668317, -7.785594118220556e-10, -1.4301497183669287e-10, 8.026890081220342, 49.976149315085394, 49.976149315074046, 49.976149315033325, 49.976149315009636, 49.97614931500597, 49.976149314983815, 49.97614931490246, 49.97614931500176, 49.97614931501379, 49.976149315038995, 8.026890081518959, 300.78324268781233, 200.44711666563646]
start = 0
end = P1.I
x_t = np.array(x[start:end])
start = end
end += P1.I
x_s1 = np.array(x[start:end])
start = end
end += P1.I
x_s2 = np.array(x[start:end])
start = end
end += P1.I
x_g = np.array(x[start:end])
start = end
end += P1.I
x_b = np.array(x[start:end])

I = np.array(range(1, P1.I+1))

fig = plt.figure()
ax_t = fig.add_subplot(2, 3, 1)
ax_t.set_ylabel("ターボ式冷凍機の熱生成量", fontname="Noto Serif CJK JP")
ax_t.set_xlabel("時刻", fontname="Noto Serif CJK JP")
# ax_t.set_yscale('log')
ax_t.bar(I, x_t)

ax_s1 = fig.add_subplot(2, 3, 2)
ax_s1.set_ylabel("蒸気吸収式冷凍機1の熱生成量", fontname="Noto Serif CJK JP")
ax_s1.set_xlabel("時刻", fontname="Noto Serif CJK JP")
# ax_1.set_yscale('log')
ax_s1.bar(I, x_s1)

ax_s2 = fig.add_subplot(2, 3, 3)
ax_s2.set_ylabel("蒸気吸収式冷凍機2の熱生成量", fontname="Noto Serif CJK JP")
ax_s2.set_xlabel("時刻", fontname="Noto Serif CJK JP")
# ax_s2.set_yscale('log')
ax_s2.bar(I, x_s2)

ax_g = fig.add_subplot(2, 3, 4)
ax_g.set_ylabel("ガスタービンのガス消費量", fontname="Noto Serif CJK JP")
ax_g.set_xlabel("時刻", fontname="Noto Serif CJK JP")
# ax_g.set_yscale('log')
ax_g.bar(I, x_g)
ax_g.bar(I, x_g)

ax_b = fig.add_subplot(2, 3, 5)
ax_b.set_ylabel("ボイラーのガス消費量", fontname="Noto Serif CJK JP")
ax_b.set_xlabel("時刻", fontname="Noto Serif CJK JP")
# ax_b.set_yscale('log')
ax_b.bar(I, x_b)
ax_b.bar(I, x_b)

plt.tight_layout()
plt.show()