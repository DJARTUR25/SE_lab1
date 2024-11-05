import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def f1(gamma):
  def rhs(t, X):
    x,y = X
    return [y , -gamma*y-x**3-x**2+2*x]
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
    plt.plot(x1, y1, linestyle='-.', color='red')                               # draws the separatrix from point (x1, y1); its view: --- and red color (because unstable)
  else:
    sol1 = solve_ivp (rhs, t, [x, y] - (h*v1), method='RK45', rtol = 1e-12)     # if 1st element negative:
    x1, y1 = sol1.y
    plt.plot(x1, y1, linestyle='-.', color='red')

  if ( v1[0] > 0 ) and ( v1[1] > 0 ):                                           # check, is the first element of eigvectors positive
    sol2 = solve_ivp(rhs, t1, [x, y] + (h*(-v1)), method='RK45', rtol = 1e-12)      # start solution from point near start point offset in the direction of eigvector
    x2, y2 = sol2.y                                                             # starting point
    plt.plot(x2, y2, linestyle='-.', color='red')                               # draws the separatrix from point (x1, y1); its view: --- and red color (because unstable)
  else:
    sol2 = solve_ivp (rhs, t1, [x, y] - (h*(-v1)), method='RK45', rtol = 1e-12)     # if 1st element negative:
    x2, y2 = sol2.y
    plt.plot(x2, y2, linestyle='-.', color='red')

  if ( v2[0] > 0 ) and ( v2[1] > 0 ):                                                     # same actions, but for the 2nd eigvector
    sol3 = solve_ivp (rhs, t, [x, y] + (h*(v2)), method='RK45', rtol = 1e-12)
    x3, y3 = sol3.y
    plt.plot(x3, y3, linestyle='-.', color='red')
  else:
    sol3 = solve_ivp (rhs, t,[x, y] - (h*(v2)), method='RK45', rtol = 1e-12)
    x3, y3 = sol3.y
    plt.plot(x3, y3, linestyle='-.', color='red')

  if ( v2[0] > 0 ) and ( v2[1] > 0 ):
    sol4 = solve_ivp (rhs, t1, [x, y] + (h*(-v2)), method='RK45', rtol = 1e-12)
    x4, y4 = sol4.y
    plt.plot(x4, y4, linestyle='-.', color='red')
  else:
    sol4 = solve_ivp (rhs, t1, [x, y] - (h*(-v2)), method='RK45', rtol = 1e-12)
    x4, y4 = sol4.y
    plt.plot(x4, y4, linestyle='-.', color='red')

  return sol1, sol2, sol3, sol4



def plot_plane(rhs, limits): # draws a vector field
  plt.figure(figsize=(13, 10))
  xlims, ylims = limits
  plt.xlim(xlims[0], xlims[1])    # the borders of field in graphic
  plt.ylim(ylims[0], ylims[1])
  xs, ys, U, V = eq_quiver(rhs, limits)   # generation of coordinates x, y
  plt.quiver(xs, ys, U, V, alpha = 1.)


# MAIN

gamma = 0.
rhs = f1(gamma)

plot_plane(rhs, ([-4., 4.], [-4., 4.]))

plt.scatter(-2, 0, color='black')
plt.scatter(1, 0, color='black')
plt.scatter(0, 0, color='red', marker = 'x')

t = [0., 10.]
t1 = [0., -10.]

sol01_in = solve_ivp (rhs, t, (-2.5, 0.), method='RK45', rtol = 1e-12)         # left center
x01_in, y01_in = sol01_in.y
plt.plot(x01_in, y01_in, color = 'green')

sol01_ag = solve_ivp (rhs, t1, (-2.5, 0.), method = 'RK45', rtol = 1e-12)
x01_ag, y01_ag = sol01_ag.y
plt.plot(x01_ag, y01_ag, color = 'green')

sol02_in = solve_ivp (rhs, t, (1.2, 0.), method='RK45', rtol = 1e-12)          # right center
x02_in, y02_in = sol02_in.y
plt.plot(x02_in, y02_in, color = 'green')

sol02_ag = solve_ivp (rhs, t1, (1.2, 0.), method = 'RK45', rtol = 1e-12)
x02_ag, y02_ag = sol02_ag.y
plt.plot(x02_ag, y02_ag, color = 'green')

sol03_in = solve_ivp (rhs, t, (-2.65, 0.), method = 'RK45', rtol = 1e-12)      # one more trajectory for left center
x03_in, y03_in = sol03_in.y
plt.plot(x03_in, y03_in, color = 'green')

sol03_ag = solve_ivp (rhs, t1, (-2.65, 0.), method = 'RK45', rtol = 1e-12)      # one more trajectory for left center
x03_ag, y03_ag = sol03_ag.y
plt.plot(x03_ag, y03_ag, color = 'green')

sol04_in = solve_ivp (rhs, t, (1.4, 0.), method = 'RK45', rtol = 1e-12)      # one more trajectory for right center
x04_in, y04_in = sol04_in.y
plt.plot(x04_in, y04_in, color = 'green')

sol04_ag = solve_ivp (rhs, t1, (1.4, 0.), method = 'RK45', rtol = 1e-12)      # one more trajectory for right center
x04_ag, y04_ag = sol04_ag.y
plt.plot(x04_ag, y04_ag, color = 'green')

sol5_in = solve_ivp (rhs, t, (3., 0.), method = 'RK45', rtol = 1e-12)        # added finite trajectories
x5_in, y5_in = sol5_in.y
plt.plot(x5_in, y5_in, color = 'blue')

sol5_ag = solve_ivp (rhs, t1, (3., 0.), method = 'RK45', rtol = 1e-12)        # added finite trajectories
x5_ag, y5_ag = sol5_ag.y
plt.plot(x5_ag, y5_ag, color = 'blue')

sol6_in = solve_ivp (rhs, t, (2., 0.), method = 'RK45', rtol = 1e-12)
x6_in, y6_in = sol6_in.y
plt.plot(x6_in, y6_in, color = 'blue')

sol6_ag = solve_ivp (rhs, t1, (2., 0.), method = 'RK45', rtol = 1e-12)
x6_ag, y6_ag = sol6_ag.y
plt.plot(x6_ag, y6_ag, color = 'blue')

sol7_in = solve_ivp (rhs, t, (1.75, 0.), method = 'RK45', rtol = 1e-12)
x7_in, y7_in = sol7_in.y
plt.plot(x7_in, y7_in, color = 'blue') 

sol7_ag = solve_ivp (rhs, t1, (1.75, 0.), method = 'RK45', rtol = 1e-12)
x7_ag, y7_ag = sol7_ag.y
plt.plot(x7_ag, y7_ag, color = 'blue') 

sep1, sep2, sep3, sep4 = plot_separatrix(rhs, 0., 0., gamma) # separatrices for central sadlle
# plt.show()


# temporary implementations x(t)


if (gamma != 0.):
  plt.tight_layout()
  plt.show()
else: 
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
  plt.plot(sol01_in.t, sol01_in.y[0], color = 'green')
  plt.plot(sol02_in.t, sol02_in.y[0], color = 'green')
  plt.xlabel('t')
  plt.ylabel('x(t)')
  plt.title('Closed trajectories')

  # 3rd cell: DOUBLE LIMIT TRAJECTORIES
  plt.subplot(3, 1, 3)
  plt.axis([0, 30, -5, 3])

  sol_dl2 = solve_ivp(rhs, [0., 100.], (-0.0001, 0.), method = 'RK45', rtol = 1e-12)     # for right separatrix
  x, y = sol_dl2.y
  plt.plot (sol_dl2.t, sol_dl2.y[0], color = 'red')

  plt.xlabel('t')
  plt.ylabel('x(t)')
  plt.title('Double Limit trajectory')
  
  plt.tight_layout()
  plt.show()