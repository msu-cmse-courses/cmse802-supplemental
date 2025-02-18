### INSTRUCTOR ANSWER 

# Claude Sonnet 3.5

from IPython.display import display, clear_output
import numpy as np
import matplotlib.pyplot as plt
import time

def show_animation(delay=0.5):
    """Creates smooth animation for gradient descent visualization.
    
    Args:
        delay (float): Time in seconds between animation frames
    """
    fig = plt.gcf()
    time.sleep(delay)
    clear_output(wait=True)
    display(fig)
    fig.clear()

def f1(x):
    """Quadratic function f(x) = 3x² - 10x + 4
    
    Args:
        x (float or np.array): Input value(s)
    
    Returns:
        float or np.array: Function value(s)
    """
    return 3*x**2 - 10*x + 4

def grad_f1(x):
    """Derivative of f1: f'(x) = 6x - 10
    
    Args:
        x (float): Input value
    
    Returns:
        float: Gradient value
    """
    return 6*x - 10

def grad_descent(x0, eta, tolerance, max_iter):
    """Performs gradient descent optimization.
    
    Args:
        x0 (float): Initial guess
        eta (float): Learning rate
        tolerance (float): Convergence threshold
        max_iter (int): Maximum number of iterations

    Returns:
        tuple: (x, f(x), number of iterations)
      
    Example:
        x_min, f_min, n_iter = grad_descent(5, 0.1, 1e-4, 100)
    
    """
        
    current_x = x0
    iter_count = 0
    
    while True:

        new_x = current_x - eta * grad_f1(current_x)
      
        if abs(current_x - new_x) <= tolerance:
            current_x = new_x           
            break

        current_x = new_x
        iter_count += 1
        

        if iter_count >= max_iter:
            print('Warning: Reached maximum iterations without convergence')
            break
            
    return current_x, f1(current_x), iter_count


def grad_descent_animation(x0, eta, tolerance, max_iter, show_animation_flag=True):
    """Performs gradient descent optimization.
    
    Args:
        x0 (float): Initial guess
        eta (float): Learning rate
        tolerance (float): Convergence threshold
        max_iter (int): Maximum number of iterations
        show_animation_flag (bool): Whether to show animation
    
    Returns:
        list: History of x values during optimization

    Example:
        x_history = grad_descent_animation(5, 0.1, 1e-4, 100)
    
    """
    x = np.linspace(-10, 10, 1000)
    current_x = x0
    x_history = [current_x]
    iter_count = 0
    
    while True:
        # Compute new position
        new_x = current_x - eta * grad_f1(current_x)
        x_history.append(new_x)
        
        # Check convergence
        if abs(current_x - new_x) <= tolerance:
            print(f'Found local minimum at x = {new_x:.6f}')
            print(f'Number of iterations: {iter_count}')
            break
            
        # Update position
        current_x = new_x
        iter_count += 1
        
        # Check iteration limit
        if iter_count >= max_iter:
            print('Warning: Reached maximum iterations without convergence')
            break
            
        # Visualization
        if show_animation_flag:
            plt.figure(0, figsize=(10, 6))
            plt.plot(x, f1(x), 'b-', lw=2, label='f(x)')
            plt.plot(x_history, f1(np.array(x_history)), 
                    'ro-', label='Gradient descent path')
            plt.ylim(-10, 100)
            plt.xlim(-4, 8)
            plt.title(f'Iteration {iter_count}: x = {new_x:.3f}')
            plt.legend()
            plt.grid(True)
            show_animation(0.5)
    
    return x_history