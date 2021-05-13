import P1
def main():
    x = []
    y = []
    f = [0]*P1.P
    f2 = [0]*P1.P
    g = [0]*P1.M
    h = [0]*int(P1.Q)
    x = P1.openfile("P1_solution_x.txt")
    y = []
    for n in range(P1.N_x):
        if x[n] < 1.0E-10:
            y.append(0.0)
        else:
            y.append(1.0)

    #evaluation
    f, g, h = P1.evaluation(x, y, f, g, h)
    V2 = 0.0
    V2, f2 = P1.evaluate_f(x)
    #output
    print(x)
    print(y)
    for p in range(P1.P):
        print("f%d = %.10g " % (p+1, f[p]))
        print("f%d = %.10g " % (p+1, f2))
    
    V = 0.0
    
    for m in range(P1.M):
        print("g%d = %.10g" % (m+1, g[m]))
        if g[m] > 0.0:
            V += g[m]
    
    for q in range(P1.Q):
        print("h%d = %.10g" % (q+1, h[q]))
        abs(q)
        V += abs(h[q])
    
    #check feasibility
    print('Sum of violation = {:.10g}'.format(V))
    print('Sum of violation = {:.10g}'.format(V2))

    print("Tolerance = {:.2g} ".format(P1.eps[0]))
    if P1.checkFeasibility(x, y):
        print("Input solution is feasible.")
    else:
        print("Input solution is infeasible.")
    

if __name__ == "__main__":
    main()
