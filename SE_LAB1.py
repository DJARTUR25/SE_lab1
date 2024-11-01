import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def f1(gamma):
  def rhs(t, X):
    x,y = X
    return [gamma + y , -x**3-x**2+2*x]
  return rhs


def sobs(x, y, gamma):

  J = np.array([[0, 1], [3*x**2+2*x, -gamma]])  # jacobian

  lambdas, vectors = np.linalg.eig(J)
  return lambdas, vectors  # eigvalues and eigvectors


def eq_quiver(rhs, limits, N=16):
  xlims, ylims = limits
  xs = np.linspace(xlims[0], xlims[1], N)
  ys = np.linspace(ylims[0], ylims[1], N)
  U = np.zeros((N,N))
  V = np.zeros((N,N))

  for i,y in enumerate(ys):
    for j,x in enumerate(xs):
      vfield = rhs(0., [x,y])
      u,v = vfield
      U[i][j] = u
      V[i][j] = v
  return xs, ys, U, V


def plot_separatrix(rhs, x, y, gamma):         # only one saddle => two separatrices
  lambdas, vectors = sobs(x, y, gamma)
  v1= vectors[:, 0]           # first eigvector
  v2 = vectors[:, 1]          # second one
  h = 0.01                    # step
  t = [0, 10]                 # in time
  t1 = [0, -10]               # againt time
  print(v1)                   # print in terminal first eigvector

  if ( v1[0] > 0 ) and ( v1[1] > 0 ):                                           # check, is the first element of eigvectors positive
    sol1 = solve_ivp(rhs, t, [x, y] + (h*v1), method='RK45', rtol = 1e-12)      # start solution from point near start point offset in the direction of eigvector
    x1, y1 = sol1.y                                                             # starting point
    plt.plot(x1, y1, linestyle='--', color='red')                               # draws the separatrix from point (x1, y1); its view: --- and red color (because unstable)
  else:
    sol1 = solve_ivp (rhs, t, [x, y] - (h*v1), method='RK45', rtol = 1e-12)     # if 1st element negative:
    x1, y1 = sol1.y
    plt.plot(x1, y1, linestyle='--', color='red')

  if ( v1[0] > 0 ) and ( v1[1] > 0 ):                                           # check, is the first element of eigvectors positive
    sol2 = solve_ivp(rhs, t, [x, y] + (h*v1), method='RK45', rtol = 1e-12)      # start solution from point near start point offset in the direction of eigvector
    x2, y2 = sol2.y                                                             # starting point
    plt.plot(x2, y2, linestyle='--', color='red')                               # draws the separatrix from point (x1, y1); its view: --- and red color (because unstable)
  else:
    sol2 = solve_ivp (rhs, t, [x, y] - (h*v1), method='RK45', rtol = 1e-12)     # if 1st element negative:
    x2, y2 = sol2.y
    plt.plot(x2, y2, linestyle='--', color='red')

  if (v2[0]>0 and v2[1]>0):                                                     # same actions, but for the 2nd eigvector
    sol3 = solve_ivp (rhs, t1, [x, y] + (h*(v2)), method='RK45', rtol = 1e-12)
    x3, y3 = sol3.y
    plt.plot(x3, y3, linestyle='--', color='red')
  else:
    sol3 = solve_ivp (rhs, t1,[x, y] - (h*(v2)), method='RK45', rtol = 1e-12)
    x3, y3 = sol3.y
    plt.plot(x3, y3, linestyle='--', color='red')

  if (v2[0]>0 and v2[1]>0):
    sol4 = solve_ivp (rhs, t1, [x, y] + (h*(-v2)), method='RK45', rtol = 1e-12)
    x4, y4 = sol4.y
    plt.plot(x4, y4, linestyle='--', color='red')
  else:
    sol4 = solve_ivp (rhs, t1,[x, y] - (h*(-v2)), method='RK45', rtol = 1e-12)
    x4, y4 = sol4.y
    plt.plot(x4, y4, linestyle='--', color='red')

  return sol1, sol2, sol3, sol4



def plot_plane(rhs, limits): # draws a vector field
  plt.figure(figsize=(8, 6))
  xlims, ylims = limits
  plt.xlim(xlims[0], xlims[1])    # the borders of field in graphic
  plt.ylim(ylims[0], ylims[1])
  xs, ys, U, V = eq_quiver(rhs, limits)   # generation of coordinates x, y
  plt.quiver(xs, ys, U, V, alpha = 0.8)


# MAIN

gamma = 0.
rhs = f1(gamma)

plot_plane(rhs, ([-4., 4.], [-4., 4.]))

plt.scatter(-2, 0, color='black')
plt.scatter(1, 0, color='black')
plt.scatter(0, 0, color='red', marker = 'x')

t = [0., 20.]
t1 = [0., -20.]

sol01 = solve_ivp (rhs, t, (-2.5, 0.), method='RK45', rtol = 1e-12)         # left center
x01, y01 = sol01.y
plt.plot(x01, y01, color = 'green')

sol02 = solve_ivp (rhs, t, (1.2, 0.), method='RK45', rtol = 1e-12)          # right center
x02, y02 = sol02.y
plt.plot(x02, y02, color = 'green')

sol03 = solve_ivp (rhs, t, (-2.65, 0.), method = 'RK45', rtol = 1e-12)      # one more trajectory for left center
x03, y03 = sol03.y
plt.plot(x03, y03, color = 'green')

sol04 = solve_ivp (rhs, t, (1.4, 0.), method = 'RK45', rtol = 1e-12)      # one more trajectory for right center
x04, y04 = sol04.y
plt.plot(x04, y04, color = 'green')

sol5 = solve_ivp (rhs, t, (3., 0.), method = 'RK45', rtol = 1e-12)        # added finite trajectories
x5, y5 = sol5.y
plt.plot(x5, y5, color = 'blue')

sol6 = solve_ivp (rhs, t, (2., 0.), method = 'RK45', rtol = 1e-12)
x6, y6 = sol6.y
plt.plot(x6, y6, color = 'blue')

sol7 = solve_ivp (rhs, t, (1.75, 0.), method = 'RK45', rtol = 1e-12)
x7, y7 = sol7.y
plt.plot(x7, y7, color = 'blue') 

sep1, sep2, sep3, sep4 = plot_separatrix(rhs, 0., 0., gamma) # separatrices for central sadlle
# plt.show()


# temporary implementations x(t)

plt.figure (figsize = (8, 6))

# 1st cell: STATES OF EQUILIBRIUM
plt.subplot (3, 1, 1)
plt.axis([0, 5, -5, 5])
plt.axhline(y = 0, color = 'blue')
plt.axhline(y = -2, color = 'blue')
plt.axhline(y = 1, color = 'blue') 
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('States of equilibrium')


# 2nd cell: CLOSED TRAJECTORIES
plt.subplot(3, 1, 2)
plt.axis([0, 10, -10, 10])
plt.plot(sol01.t, sol01.y[0], color = 'green')
plt.plot(sol02.t, sol02.y[0], color = 'green')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('Closed trajectories')

# 3rd cell: DOUBLE LIMIT TRAJECTORIES
plt.subplot(3, 1, 3)
plt.axis([0, 10, -5, 3])
plt.plot(sep1.t, sep1.y[0], color = 'red')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('Double Limit trajectory')


plt.tight_layout()
plt.show()