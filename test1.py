###########
#running coupling algorithm, use analitical expressions
import numpy as np
import sympy 
from scipy import integrate
import matplotlib.pyplot as plt
import math
from numba import jit

@jit
def solution(Q,b,bij,alpha0):
    a1 = []
    a2 = []
    a3 = []
    h1 =0
    h2 =0
    h3 =0
    for i in range(0,len(Q)):
        for j in range(0,3): #problem when j = 3 we have log of a negative quantity
            h3 = h3 + (bij[0][j]/(4*3.14*b[0]))*np.log(1-(b[j]/(2*3.14))*alpha0[j]*np.log(Q[i]/Q[0]))
            h2 = h2+ (bij[1][j]/(4*3.14*b[1]))*np.log(1-(b[j]/(2*3.14))*alpha0[j]*np.log(Q[i]/Q[0]))
            h1 = h1+  (bij[2][j]/(4*3.14*b[2]))*np.log(1-(b[j]/(2*3.14))*alpha0[j]*np.log(Q[i]/Q[0]))
        g3 = alpha0[0]**(-1)-b[0]*math.log(Q[i]/Q[0])+ h3 
        a3.append(g3)
        h1 =0
        g2 = alpha0[0]**(-1)-b[1]*math.log(Q[i]/Q[0])+ h2
        a2.append(g2)
        h2 =0
        g1 = alpha0[0]**(-1)-b[2]*math.log(Q[i]/Q[0])+ h1 
        a1.append(g1)
        h3 =0
    return a3,a2,a1
@jit
def solution4(Q,b,bij,alpha0):
    a1 = []
    a2 = []
    a3 = []
    a4 = []
    h1 =0
    h2 =0
    h3 =0
    h4 =0
    for i in range(0,len(Q)):
        for j in range(0,4): #problem when j = 3 we have log of a negative quantity
            h4 = h4 + (bij[0][j]/(4*3.14*b[0]))*np.log(1-(b[j]/(2*3.14))*alpha0[j]*np.log(Q[i]/Q[0]))
            h3 = h3 + (bij[1][j]/(4*3.14*b[1]))*np.log(1-(b[j]/(2*3.14))*alpha0[j]*np.log(Q[i]/Q[0]))
            h2 = h2+ (bij[2][j]/(4*3.14*b[2]))*np.log(1-(b[j]/(2*3.14))*alpha0[j]*np.log(Q[i]/Q[0]))
            h1 = h1+  (bij[3][j]/(4*3.14*b[3]))*np.log(1-(b[j]/(2*3.14))*alpha0[j]*np.log(Q[i]/Q[0]))
        g4 = alpha0[0]**(-1)-b[0]*math.log(Q[i]/Q[0]) + h4
        a4.append(g4)
        g3 = alpha0[1]**(-1)-b[1]*math.log(Q[i]/Q[0])+ h3 
        a3.append(g3)
        h1 =0
        g2 = alpha0[2]**(-1)-b[2]*math.log(Q[i]/Q[0])+ h2
        a2.append(g2)
        h2 =0
        g1 = alpha0[3]**(-1)-b[3]*math.log(Q[i]/Q[0])+ h1 
        a1.append(g1)
        h3 =0
    return a4, a3,a2,a1


alpha0 = np.array([0.1184,0.033819,0.010168])
b = np.array([-7,-19/6,41/10])
bij = np.array([[-26,9/2,11/10],[12,35/6,9/10],[44/5,17/10,199/50]])
Q = np.linspace(90,1e9,1000)

b1 = np.array([-7,-8/3,-2,11/2])
b1ij = np.array([[-26,9/2,9/2,1/2],[12,37/3,6,3/2],[12,6,31,27/2],[4,9/2,81/2,61/2]])

b2 = np.array([-7,-2,-2,7])
#b = np.array([-7,-8/3,-2,11/2])
b2ij = np.array([[-26,9/2,9/2,1/2],[12,31,6,27/2],[12,6,31,27/2],[4,81/2,81/2,115/2]])

b3 = np.array([10/3,26/3,26/3])
b3ij = np.array([[4447/6,249/2,249/2],[1245/3,779/3,48],[1245/2,48,779/3]])
                            

sol3, sol2, sol1 = solution(Q,b,bij,alpha0)





####run from 10^7 to M1 scale



@jit
def run(sol3,sol2,sol1,b,bij,b1,b1ij,b2,b2ij,b3,b3ij):
    M_1 = []
    M_2 = []
    M_3 = []
    M_U = []
    for M1 in range(int(5e10),int(5e13),int(1e9)):
        alpha0 = np.array([sol3[999]**(-1), sol2[999]**(-1), sol1[999]**(-1)])
        Q = np.linspace(1e9, M1, 1000)
        sol3,sol2,sol1 = solution(Q,b,bij,alpha0)
        
        ####implement first matching condition: ##generate random a2R and compute a1Y 
        for a2R in range(20,80):
            a1x = (sol1[999]-(3/5)*(a2R-(1/(6*3.14))))*(5/2)
            alpha0 = np.array([ sol3[999]**(-1),sol2[999]**(-1),a2R**(-1),a1x**(-1)])
            
            #run from M1 to M2 scale
            
            for M2 in range(int(1e11),int(1e16),int(1e9)):
                print(M2)
                if(M2 > M1):
                    Q = np.linspace(M1,M2,1000)
                    sol4,sol3,sol2,sol1  = solution4(Q,b1,b1ij,alpha0)
                    
                    
                    #no matching in M2 only change beta and the run again until M3 ###impose a2R = a2L at the end, impose a2r =a2L
                    if (sol3[999]**(-1) -0.005 < sol2[999]**(-1) < sol3[999]**(-1) + 0.005):
                        alpha0 = np.array([sol4[999]**(-1),sol3[999]**(-1),sol2[999]**(-1),sol1[999]**(-1)])
                        #print("%f %f %f %f" % (alpha0[0],alpha0[1],alpha0[2],alpha0[3]))
                        for M3 in range(int(1e11), int(1e16),int(1e9)):
                            if (M3 > M2):
                                Q = np.linspace(M2, M3,1000) #max M3 5e15
                                sol4,sol3,sol2,sol1 = solution4(Q,b2,b2ij,alpha0)
                                #impose matching at M3
                                a4 = sol4[999] + (1/(12*3.14))
                                if (sol1[999]**(-1) -0.005 < a3**(-1) -(1/(4*3.14)) < sol1[999]**(-1) + 0.005):
                                    #run until unification set now MU0 10^16
                                    alpha0 = np.array([a4**(-1),sol3[999]**(-1),sol2[999]**(-1)])
                                    for MU in range(int(1e14),int(1e17),int(1e9)):
                                        if(MU > M3):
                                            Q = np.linspace(M3, MU, 1000)
                                            sol3, sol2, sol1 = solution(Q, b3, b3ij,alpha0)
                                            if (sol3[999]**(-1) -0.005 < sol2[999]**(-1)< sol3[999]**(-1) + 0.005):
                                                M_1.append(M1)
                                                M_2.append(M2)
                                                M_3.append(M3)
                                                M_U.append(MU)
                                                M_1 = np.asarray(M_1)
                                                M_2 = np.asarray(M_2)
                                                M_3 = np.asarray(M_3)
                                                M_U = np.asarray(M_U)
    return M_1,M_2,M_3,M_4
#fig, ax = plt.subplots()
                                            
m1,m2,m3,mu = run(sol3,sol2,sol1,b,bij,b1,b1ij,b2,b2ij,b3,b3ij)
np.savetxt("M1.txt",m1)
np.savetxt("M2.txt",m2)
np.savetxt("M3.txt",m3)
np.savetxt("MU.txt",mu)

#plt.plot(Q,sol2,'b',label="ss")
#ax.set_xscale("log")
#ax.set_yscale("log")
#plt.show()

####
#in the final version generate random a3 and impose all the condition otherwise the program cannot run
