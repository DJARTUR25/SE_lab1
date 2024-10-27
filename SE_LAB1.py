import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def f1(alpha, beta):    # function equal system of differenctional 
  def rhs(t, X):
    x,y = X
    return [y, -2*alpha - x*beta]
  return rhs


def jacobian(alpha, beta):    # function of matrix of Jacobi
  return np.array([[0,1], [-alpha, -2*beta]])


def eq_quiver(rhs, limits, N=16):   # function creates the limits of linear limits
  xlims, ylims = limits
  xs = np.linspace(xlims[0], xlims[1], N)
  ys = np.linspace(ylims[0], ylims[1], N)
  U = np.zeros((N,N))
  V = np.zeros((N,N))

  for i,y in enumerate(ys):
    for j,x in enumerate(xs):
      vfield = rhs(0.0, [x,y])
      u, v = vfield
      U[i][j] = u
      V[i][j] = v
  return xs, ys, U, V

def plotonPlane(rhs, limits):   # function draws the phase plate
  plt.close()
  xlims, ylims = limits
  plt.xlim(xlims[0], xlims[1])
  plt.ylim(ylims[0], ylims[1])
  xs, ys, U, V = eq_quiver(rhs, limits)
  plt.quiver(xs, ys, U, V, alpha=0.8)

alpha = 1.0     # main, makes the constants
beta = 1.0
rhs = f1(alpha, beta)
plotonPlane(rhs, [(-5., 5.), (-5., 5.)])
sol1 = solve_ivp(rhs, [0., 100.], (0., 4.), method='RK45', rtol=1e-12)
x1, y1 = sol1.y
plt.plot(x1,y1,'b-')
sol2 = solve_ivp(rhs, [0., -100.], (0., 4.), method='RK45', rtol=1e-12)
x2, y2 = sol2.y
plt.plot(x2,y2,'b-')
plt.show()