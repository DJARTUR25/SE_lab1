import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def f1(delta, gamma):    # function equal system of differenctional 
  def rhs(t, X):
    x,y = X
    return [y, -2*delta*y - x*gamma]
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
  elif np.all(real > 0) and np.all(imag == 0):    # if real parts of eigenvalues are more than 0 and imagine parts do not exist => unstable node
    return "Unstable node"
  elif np.any (real < 0) and np.any (real > 0) and np.all(imag == 0):   # if real parts of a different signs and imagine parts do not exist => saddle
    return "Saddle"
  elif np.all(real < 0) and np.all(imag != 0):    # if real parts of eigenvalues are less than 0 and imagine parts exists => stable focus
    return "Stable focus"
  elif np.all(real > 0) and np.all(imag != 0):    # if real parts of eigenvalues are more than 0 and imagine parts exists => unstable focus
    return "Unstable foсus"
  elif np.all(np.isclose(real, 0.0)) and np.all(imag != 0):   # if eigenvalues are purely imaginary => center (or difficult focus, more research is needed)
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
  
def plot_phase_portrait(ax, rhs, delta, gamma, limits, point_color, traj_color):
    xlims, ylims = limits
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xlabel('X')
    ax.set_ylabel("Ẋ")

    eig_vals, eig_vecs = calc_eigvalues(delta, gamma)
    eq_type = get_type(eig_vals)

    eig1_str = print_complex_numbers(eig_vals[0])
    eig2_str = print_complex_numbers(eig_vals[1])

    ax.set_title(f"{eq_type}\nСобственные числа: λ₁={eig1_str}, λ₂={eig2_str}")

    N = 20 
    x = np.linspace(xlims[0], xlims[1], N)
    y = np.linspace(ylims[0], ylims[1], N)
    X, Y = np.meshgrid(x, y)
    U, V = np.zeros_like(X), np.zeros_like(Y)
    for i in range(N):
        for j in range(N):
            dx, dy = rhs(0, [X[i, j], Y[i, j]])
            U[i, j] = dx
            V[i, j] = dy
    # Normalixation
    M = np.hypot(U, V)
    M[M == 0] = 1
    U /= M
    V /= M
    ax.quiver(X, Y, U, V, M, pivot='mid', cmap='gray', alpha=0.4)

    ax.plot(0, 0, 'o', color=point_color, markersize=5, label = eq_type)

    initial_conditions = [
        (3, 0), (-3, 0),
        (0, 3), (0, -3),
        (3, 3), (-3, 3),
        (3, -3), (-3, -3)
    ]
    t_span = [0, 20]
    for ic in initial_conditions:
        # Траектории вперёд по времени
        sol = solve_ivp(rhs, t_span, ic, dense_output=True, max_step=0.1)
        ax.plot(sol.y[0], sol.y[1], color=traj_color, linewidth=1.5, alpha=0.9)
        # Траектории назад по времени
        sol_back = solve_ivp(rhs, [0, -20], ic, dense_output=True, max_step=0.1)
        ax.plot(sol_back.y[0], sol_back.y[1], color=traj_color, linewidth=1.5, alpha=0.9)

 

 # main function

cases = [
    {'delta': 1.2, 'gamma': 1.0, 'point_color': 'black', 'traj_color': 'blue'}, # УУ
    {'delta': -1.2, 'gamma': 1.0, 'point_color': 'black', 'traj_color': 'red'}, # НУ
    {'delta': 1.0, 'gamma': 2.0, 'point_color': 'black', 'traj_color': 'purple'}, # УФ
    {'delta': -1.0, 'gamma': 2.0, 'point_color': 'black', 'traj_color': 'orange'}, # НФ
    {'delta': 0.0, 'gamma': 1.0, 'point_color': 'black', 'traj_color': 'gray'},   # Ц
    {'delta': 0.5, 'gamma': -0.5, 'point_color': 'black', 'traj_color': 'green'},  # С
]

fig, axes = plt.subplots(2, 3, figsize=(12, 10))
axes = axes.flatten()


for idx, case in enumerate(cases):
    ax = axes[idx]
    rhs = f1(case['delta'], case['gamma'])
    limits = [(-5, 5), (-5, 5)]
    plot_phase_portrait(ax, rhs, case['delta'], case['gamma'], limits, point_color=case['point_color'], traj_color=case['traj_color'])

plt.tight_layout()
plt.show()