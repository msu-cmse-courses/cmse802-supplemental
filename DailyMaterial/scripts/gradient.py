import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output

def show_animation():
  fig = plt.gcf()
  clear_output(wait=True) # Clear output for dynamic display
  display(fig)      # Reset display
  fig.clear()       # Prevent overlapping and layered plots

def f1(x):
  """Function to minimize"""
  return 3*x**2 - 10*x + 4

def grad_f1(x):
  """Gradient of function to minimize"""
  return 6*x - 10

def grad_descent(x0, eta, p, maxIter, show):
  """Finding minimum using gradient descent.
  Args:
    x0 (int) -- Starting point for gradient descent
    eta (float) -- Step size (a.k.a learning rate)
    p (float) -- Chosen precision
    maxIter (int) -- Maximum number of iterations
    show (int) -- show animation (1) or not (0)
  Returns:
    iterative_mins (list) -- a list of minima over the iterations
  """
  x = np.linspace(-10,10,1000)
  former_min = x0
  iterative_mins = [former_min]
  iterCounter = 1

  while True:
    new_min = former_min - eta * grad_f1(former_min)

    iterative_mins.append(new_min)
    if abs(former_min - new_min) <= p:
      print('Local min of function is %f' %new_min)
      print('Number of iterations: %d' %iterCounter)
      break
    else:
      former_min = new_min

    if iterCounter == maxIter:
      print('Local min not reached')
      break
    else:
      iterCounter += 1     

    if show:
      plt.figure(0)
      plt.plot(x, f1(x), lw=3)
      plt.ylim(-10,100)
      plt.xlim(-4,8)
      plt.title(f'iter={str(iterCounter)}, min={str(round(new_min,3))}')
      plt.plot(iterative_mins, f1(np.array(iterative_mins)), marker='o')
      show_animation()
    
  return iterative_mins