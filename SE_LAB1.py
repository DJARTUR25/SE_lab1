import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def f1(delta, gamma):    # function equal system of differenctional 
  def rhs(t, X):
    x,y = X
    return [y, -(x**3) - (x**2) + 2*x]
  return rhs


def jacobian(delta, gamma):    # function of matrix of Jacobi
  return np.array([[0,1], [-delta, -2*gamma]])


def calc_eigvalues (delta, gamma):     # function that calculating eigenvalues and eigenvectors 
  vals, vectors = np.linalg.eig(jacobian(delta,gamma))
  return vals, vectors

def get_type(vals):     # determines what type of equilibrium a given state has
  real = vals.real
  imag = vals.imag
  
  if np.all(real < 0) and np.all(imag == 0):    # if real parts of eigenvalues are less than 0 and imagine parts do not exist => stable node
    return "Stable node"
  else:  
    if np.all(real > 0) and np.all(imag == 0):    # if real parts of eigenvalues are more than 0 and imagine parts do not exist => unstable node
      return "Unstable node"
    else:
      if np.any (real < 0) and np.any (real > 0) and np.all(imag == 0):   # if real parts of a different signs and imagine parts do not exist => saddle
        return "Saddle"
      else:
        if np.all(real < 0) and np.all(imag != 0):    # if real parts of eigenvalues are less than 0 and imagine parts exists => stable focus
          return "Stable focus"
        else:
          if np.all(real > 0) and np.all(imag != 0):    # if real parts of eigenvalues are more than 0 and imagine parts exists => unstable focus
            return "Unstable foсus"
          else:
            if np.all(real == 0) and np.all(imag != 0):   # if eigenvalues are purely imaginary => center (or difficult focus, more research is needed)
              return "Center"


def print_complex_numbers(num):
  if np.iscomplex(num):
    if num.imag >= 0: 
      sign = '+'
    else:
      sign = '-'
    return f"{num.real:f} {sign} {abs(num.imag):f}i"
  else:
    return f"{num:f}"
  

def plot_phase_portrait(ax, rhs, delta, gamma, limits, fixed_point_color, trajectory_color):
  xlims, ylims = limits
  ax.set_xlim(xlims)
  ax.set_ylim(ylims)
  ax.set_xlabel('X')
  ax.set_ylabel("X'")

  selfval, selfvect =  calc_eigvalues(delta, gamma)    # calculate eigvalues and eigvectors for this case
  balance_type = get_type(selfval)
  eq1 = print_complex_numbers(selfval[0])
  eq2 = print_complex_numbers(selfval[1])
  ax.set_title(f"This equation is: {balance_type}. Eigvalues is: λ1 = {eq1}; λ2 = {eq2}")    # and print them in terminal (I hope, for now)

  N = 20
  xs = np.linspace(xlims[0], xlims[1], N)
  ys = np.linspace(ylims[0], ylims[1], N)
  X, Y = np.meshgrid(xs, ys)
  U, V = np.zeros_like(X), np.zeros_like(Y)
  for i in range(N):
    for j in range(N):
      dx, dy = rhs(0, [X[i, j], Y[i, j]])
      U[i, j] = dx
      V[i, j] = dy
  
  # normalization
  M = np.hypot(U,V)
  M[M == 0] = 1
  U = U / M
  V = V / M
  ax.quiver (X, Y, U, V, M, pivot = 'mid', cmap = 'gray', alpha = 0.8)

  ax.plot(0, 0, 'o', color = fixed_point_color, markersize = 8, label = balance_type)


cases = [
    {'delta': 1.2, 'gamma': 1.0, 'fixed_color': 'black', 'trajectory_color': 'blue'}, # УУ
    {'delta': -1.2, 'gamma': 1.0, 'fixed_color': 'black', 'trajectory_color': 'red'}, # НУ
    {'delta': 1.0, 'gamma': 2.0, 'fixed_color': 'black', 'trajectory_color': 'purple'}, # УФ
    {'delta': -1.0, 'gamma': 2.0, 'fixed_color': 'black', 'trajectory_color': 'orange'}, # НФ
    {'delta': 0.0, 'gamma': 1.0, 'fixed_color': 'black', 'trajectory_color': 'gray'},   # Ц
    {'delta': 0.5, 'gamma': -0.5, 'fixed_color': 'black', 'trajectory_color': 'green'},  # С
]

#main 
fig, axes = plt.subplots(nrows = 2, ncols = 3, sharex=True, sharey=True) # figsize=(12, 10))
axes = axes.flatten()

for idx, case in enumerate(cases):
    ax = axes[idx]
    rhs = f1(case['delta'], case['gamma'])
    limits = [(-5, 5), (-5, 5)]
    plot_phase_portrait(ax, rhs, case['delta'], case['gamma'], limits, fixed_point_color=case['fixed_color'], trajectory_color=case['trajectory_color'])




plt.tight_layout()
plt.show()