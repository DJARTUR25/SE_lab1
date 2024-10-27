import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def f1(alpha, beta):    # function equal system of differenctional 
  def rhs(t, X):
    x,y = X
    return [y, -(x**3) - (x**2) + 2*x]
  return rhs


def jacobian(alpha, beta):    # function of matrix of Jacobi
  return np.array([[0,1], [-alpha, -2*beta]])


def calc_eigvalues (alpha, beta):     # function that calculating eigenvalues and eigenvectors 
  vals, vectors = np.linalg.eig(jacobian(alpha,beta))
  return vals, vectors

def get_type(vals):     # determines what type of equilibrium a given state has
  real = vals.real
  imag = vals.imag
  
  if np.all(real < 0) and np.all(imag == 0):    # if real parts of eigenvalues are less than 0 and imagine parts do not exist => stable node
    return "Stable node"
  if np.all(real > 0) and np.all(imag == 0):    # if real parts of eigenvalues are more than 0 and imagine parts do not exist => unstable node
    return "Unstable node"
  if np.any (real < 0) and np.any (real > 0) and np.all(imag == 0):   # if real parts of a different signs and imagine parts do not exist => saddle
    return "Saddle"
  if np.all(real < 0) and np.all(imag != 0):    # if real parts of eigenvalues are less than 0 and imagine parts exists => stable focus
    return "Stable focus"
  if np.all(real > 0) and np.all(imag != 0):    # if real parts of eigenvalues are more than 0 and imagine parts exists => unstable focus
    return "Unstable foсus"
  if np.all(real == 0) and np.all(imag != 0):   # if eigenvalues are purely imaginary => center (or difficult focus, more research is needed)
    return "Center"


def print_complex_numbers(num):
  if np.iscomplex(num):
    if num.imag >= 0: 
      sign = '+'
    else:
      sign = '-'
    return f"{num.real::2f} {sign} {abs(num.imag)::2f}i"
  return f"{num::2f}"
  

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