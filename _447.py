# numpy = numerical Python, implements arrays (/ matrices)
import numpy as np
# limit number of decimal places printed for floating-point numbers
np.set_printoptions(precision=3)

# scipy = scientific Python, implements operations on arrays / matrices
import scipy as sp
# linalg = linear algebra, implements eigenvalues, matrix inverse, etc
from scipy import linalg as la
# optimize = optimization, root finding, etc
from scipy import optimize as op
# signal processing = filters, LTI systems
from scipy import signal as sig

# produce matlab-style plots
import matplotlib as mpl
# increase font size on plots
mpl.rc('font',**{'size':12})
# use LaTeX to render symbols
mpl.rc('text',usetex=False)
# render animation
mpl.rc('animation',html='html5')
# animation
from matplotlib import animation as ani
# Matlab-style plotting
import matplotlib.pyplot as plt

# symbolic computation, i.e. computer algebra (like Mathematica, Wolfram Alpha)
import sympy as sym
# print symbolic expressions using LaTeX-style formatting
sym.init_printing(use_latex='mathjax')

# display(Math( latex_expression )) prints formatted math
from IPython.display import display, Math

# --
# Prof Burden provides the following functions used throughout the class:

def Jacobian(g,y,d=1e-4):
  """
  approximate derivative via finite-central-differences

  input:
    g - function - g : R^n -> R^m
    y - n array
    (optional)
    d - scalar - finite differences displacement parameter

  output:
    Dg(y) - m x n - approximation of Jacobian of g at y
  """
  # given $g:\mathbb{R}^n\rightarrow\mathbb{R}^m$:
  # $$D_y g(y)e_j \approx \frac{1}{2\delta}(g(y+\delta e_j) - g(y - \delta e_j)),\ \delta\ll 1$$
  e = np.identity(len(y))
  Dyg = []
  for j in range(len(y)):
      Dyg.append((.5/d)*(g(y+d*e[j]) - g(y-d*e[j])))
  return np.array(Dyg).T

def numerical_simulation(f,t,x,t0=0.,dt=1e-4,ut=None,ux=None,utx=None,return_u=False):
  """
  simulate x' = f(x,u)

  input:
    f : R x X x U --> X - vector field
      X - state space (must be vector space)
      U - control input set
    t - scalar - final simulation time
    x - initial condition; element of X

    (optional:)
    t0 - scalar - initial simulation time
    dt - scalar - stepsize parameter
    return_u - bool - whether to return u_

    (only one of:)
    ut : R --> U
    ux : X --> U
    utx : R x X --> U

  output:
    t_ - N array - time trajectory
    x_ - N x X array - state trajectory
    (if return_u:)
    u_ - N x U array - state trajectory
  """
  t_,x_,u_ = [t0],[x],[]

  inputs = sum([1 if u is not None else 0 for u in [ut,ux,utx]])
  assert inputs <= 1, "more than one of ut,ux,utx defined"

  if inputs == 0:
    assert not return_u, "no input supplied"
  else:
    if ut is not None:
      u = lambda t,x : ut(t)
    elif ux is not None:
      u = lambda t,x : ux(x)
    elif utx is not None:
      u = lambda t,x : utx(t,x)

  while t_[-1]+dt < t:
    if inputs == 0:
      _t,_x = t_[-1],x_[-1]
      dx = f(t_[-1],x_[-1]) * dt
    else:
      _t,_x,_u = t_[-1],x_[-1],u(t_[-1],x_[-1])
      dx = f(_t,_x,_u) * dt
      u_.append( _u )

    x_.append( _x + dx )
    t_.append( _t + dt )

  if return_u:
    return np.asarray(t_),np.asarray(x_),np.asarray(u_)
  else:
    return np.asarray(t_),np.asarray(x_)

def phase_portrait(f,xlim=(-1,+1),ylim=(-1,+1),n=0,t=1,
                   quiver=False,fs=(5,5),fig=None,ax=None):
    """
    phase portrait of a planar vector field 

    input:
    f : R x R^2 --> R^2 - planar vector field
    xlim,ylim - 2-tuple - bounds for portrait
    n - int - number of trajectories to simulate
    t - scalar - simulation duration

    (optional:)
    quiver - bool - use quiver rather than stream plot
    fs - 2-tuple - figure size

    output:
    fig,ax - figure and axis handles
    """
    if fig is not None:
        fig = plt.figure(figsize=fs)

    # phase portrait / "quiver" plot
    if ax is None:
        ax = plt.subplot(1,1,1)
    X, Y = np.meshgrid(np.linspace(xlim[0],xlim[1], 11), np.linspace(ylim[0], ylim[1], 11))
    dX,dY = np.asarray([f(0,xy).flatten() for xy in zip(X.flatten(),Y.flatten())]).T
    dX,dY = dX.reshape(X.shape),dY.reshape(Y.shape)
    if quiver:
        ax.quiver(X,Y,dX,dY)
    else:
        ax.streamplot(X,Y,dX,dY,density=2.,color=(0,0,0,.5))
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')

    for _ in range(n):
      x0 = np.random.rand(2)*[np.diff(xlim)[0],np.diff(ylim)[0]] + [xlim[0],ylim[0]]
      t_,x_ = numerical_simulation(f,t,x0)
      ax.plot(x_[:,0],x_[:,1])

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.tight_layout()
    
    return fig,ax

def nyquist_plot(L,omega=None,fs=(5,5),fig=None,ax=None):
    """
    Nyquist plot of a transfer function  

    input:
    L : C --> C - single-input single-output transformation

    output:
    fig,ax - figure and axis handles
    """
    if omega is None:
        omega = np.logspace(-2,+2,1000)
        
    Omega = L(1.j*omega)

    abs_L = np.abs(Omega)
    angle_L = np.unwrap(np.angle(Omega))*180./np.pi

    circle = np.exp(1.j*np.linspace(np.pi/2,3*np.pi/2))

    if fig is None:
        fig = plt.figure(figsize=fs)
    if ax is None:
        ax = plt.subplot(1,1,1)
    ax.grid('on'); ax.axis('equal')
    # Omega, i.e. graph of L(j omega)
    ax.plot(Omega.real,Omega.imag,'b-',label=r'$L(j\omega), \omega > 0$')
    ax.plot(Omega.real,-Omega.imag,'b--',label=r'$L(j\omega), \omega < 0$')
    # unit circle
    ax.plot(circle.real,circle.imag,'k--',zorder=-1)
    ylim = ax.get_ylim()
    ax.vlines(0,ylim[0],ylim[1],color='k',ls='--',zorder=-1)
    ax.set_ylim(ylim)
    # critical point (-1. + 0.j)
    ax.plot(-1.,0.,'ro',label=r'critical point $-1\in\mathbb{C}$')
    # axis labels
    ax.set_xlabel(r'$\operatorname{Re}\ L(j\omega)$')
    ax.set_ylabel(r'$\operatorname{Im}\ L(j\omega)$');
    
    return fig,ax

#--