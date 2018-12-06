import numpy as np
import matplotlib.pylab as plt
plt.ion()

def quadratic(x):
	return (x-1)**2
def quadratic_deriv(x):
	return 2*(x-1)

def valley(x):
	return ((x-3)**2) * ((x+3)**2 + 1)
def valley_deriv(x):
	return 2*(x-3)*((x+3)**2 + 1) + 2*(x+3)*((x-3)**2)

def gradient_descent(f, f_deriv, x_start, learning_rate=1, n_iter=10):
	x_path = [x_start]
	for i in range(n_iter+1):
		x_next = x_path[i] - learning_rate * f_deriv(x_path[i])
		x_path.append(x_next)

	f_vals = [f(x) for x in x_path]
	return x_path, f_vals

def plot(x_start, learning_rate, f, f_deriv, n_iter, save_name=None, x=np.arange(-10, 12, 0.1)):
	plt.figure(figsize=(10,8))

	y = [f(i) for i in x]
	plt.plot(x,y,label='function')
	plt.xlabel('x')
	plt.ylabel('f(x)')
	plt.title("Gradient descent")

	x_path, f_vals = gradient_descent(f, f_deriv, x_start, learning_rate=learning_rate, n_iter=n_iter)
	plt.plot(x_path, f_vals, 'p-', color='red', label=f'lr={learning_rate}, start={x_start}')
	plt.plot(x_path[0], f_vals[0], marker="+", color='green', markersize=20)
	plt.text(x_path[0], f_vals[0], 'start')
	plt.plot(x_path[-1], f_vals[-1], marker="+", color='blue', markersize=20)
	plt.text(x_path[-1], f_vals[-1], 'end')
	#for i in range(len(x_path)-1):
	#	plt.arrow(x_path[i], f_vals[i], x_path[i+1]-x_path[i], f_vals[i+1]-f_vals[i], color='red', shape='full', head_width=0.5)

	plt.legend(framealpha=0)

	if save_name is not None:
		plt.savefig(save_name, transparent=True)

def convex_plots():
	plot(-5, 1.1, quadratic, quadratic_deriv, 2, save_name='plots/convex_lrlarge.png') #lr too large
	plot(-5, 0.1, quadratic, quadratic_deriv, 10, save_name='plots/convex_lrsmall.png') #convergence
	plot(-5, 0.8, quadratic, quadratic_deriv, 10, save_name='plots/convex_bounce.png')

	plot(0.5, 0.01, valley, valley_deriv, 100, save_name='plots/nonconvex_right_lrsmall.png', x=np.arange(-5,5,0.1)) #right, small lr
	plot(0.5, 0.05, valley, valley_deriv, 10, save_name='plots/nonconvex_right_lrlarge.png', x=np.arange(-5,5,0.1))
	plot(-0.5, 0.01, valley, valley_deriv, 100, save_name='plots/nonconvex_left_lrsmall.png', x=np.arange(-5,5,0.1)) #right, small lr
	plot(-0.5, 0.05, valley, valley_deriv, 10, save_name='plots/nonconvex_left_lrlarge.png', x=np.arange(-5,5,0.1))