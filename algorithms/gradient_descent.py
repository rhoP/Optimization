# algorithms/gradient_descent.py
import numpy as np

def gradient_descent(func, initial_x, learning_rate=0.01, max_iters=100):
    x = np.array(initial_x, dtype=np.float32)
    for _ in range(max_iters):
        _, grad = evaluate_function_and_gradient(func, x)
        x = x - learning_rate * grad
    return x

