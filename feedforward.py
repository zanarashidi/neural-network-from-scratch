from utils import *
import numpy as np 
from scipy.special import expit
import random as rnd


# sigmoid function
def sigmoid(x, deriv=False):
	return expit(x) * (1 - expit(x)) if deriv else expit(x)


# softmax function
def softmax(x):
	e_x = np.exp(x - np.max(x))
	return e_x / np.sum(e_x)


def neural_network(train_data, train_labels, test_data, test_labels, digits):
	# dataset parameters
	width, height = 28, 28
	dimension = width*height
	size = len(train_data)
	num_digits = len(digits)
	train_labels = np.array(train_labels)

	# reshape training data to appropriate format: x
	train_data = train_data.reshape(-1, dimension)
	train_data = train_data.T
	x = [np.reshape(train_data[:,i], (dimension, 1)) for i in range(size)]

	# reshape training labels: y
	target_y = np.zeros((size, num_digits))
	for i in range(num_digits):
		for k in range(size):
			if train_labels[k] == digits[i]:
				target_y[k,i] = 1.0
	target_y = target_y.T
	y = [np.reshape(target_y[:,i], (num_digits, 1)) for i in range(size)]

	# reshape test data: t_x
	t_size = len(test_data)
	test_data = test_data.reshape(-1, dimension)
	test_data = test_data.T
	t_x = [np.reshape(test_data[:,i], (dimension, 1)) for i in range(t_size)]

	# hyperparameters
	num_hls = 3
	hl_size = 200
	epochs = 200
	lr = 0.01
	batch_size = 600
	num_batches = int(size / batch_size)
	rand_idx = np.arange(size)

	# weights and biases
	w = []
	b = []
	w.append(np.random.randn(dimension, hl_size).T)
	b.append(np.zeros((1, hl_size)).T)
	for i in range(1, num_hls):
		w.append(np.random.randn(hl_size, hl_size).T)
		b.append(np.zeros((1, hl_size)).T)
	w.append(np.random.randn(hl_size, num_digits).T)
	b.append(np.zeros((1, num_digits)).T)

	# logits 
	z = []
	for i in range(1, num_hls+1):
		z.append(np.zeros((hl_size, 1)))
	z.append(np.zeros((num_digits, 1)))

	# activations
	acts = []
	acts.append(np.zeros((dimension, 1)))
	for i in range(1, num_hls+1):
		acts.append(np.zeros((hl_size, 1)))
	acts.append(np.zeros((num_digits, 1)))

	# training loop
	print('Training ...\n')
	for i in range(epochs):
		# gradients
		nabla = [np.zeros(weight.shape) for weight in w]
		nabla_b = [np.zeros(bias.shape) for bias in b]

		# batch shuffle
		rnd.shuffle(rand_idx)
		temp_x = [x[k] for k in rand_idx]
		temp_y = [y[k] for k in rand_idx]
		x = temp_x
		y = temp_y

		for e in range(num_batches):
			for k in range(batch_size):

				# forward prop
				acts[0] = x[e*batch_size+k]
				for j in range(num_hls):
					z[j] = w[j].dot(acts[j]) + b[j]
					acts[j+1] = sigmoid(z[j])
				z[-1] = w[-1].dot(acts[-2]) + b[-1]
				acts[-1] = softmax(z[-1])

				# backward prop
				err = acts[-1] - y[e*batch_size+k]
				nabla_t = [np.zeros(weight.shape) for weight in w]
				nabla_t_b = [np.zeros(bias.shape) for bias in b]
				nabla_t[-1] = err.dot(acts[-2].T)
				nabla_t_b[-1] = err
				for j in range(num_hls, 0, -1):
					err = np.multiply(w[j].T.dot(err), sigmoid(z[j-1], deriv=True))
					nabla_t[j-1] = err.dot(acts[j-1].T)
					nabla_t_b[j-1] = err

				# accumulate gradients
				delta_nabla = nabla_t
				delta_nabla_b = nabla_t_b
				nabla = [n + dn for n, dn in zip(nabla, delta_nabla)]
				nabla_b = [n + dn for n, dn in zip(nabla_b, delta_nabla_b)]

			# update weights and biases
			w = [weight - (lr / batch_size) * dw for weight, dw in zip(w, nabla)]
			b = [bias - (lr / batch_size) * db for bias, db in zip(b, nabla_b)]

		# validation accuracy
		corr_labels = 0
		for k in range(t_size):
			acts[0] = t_x[k]
			for j in range(num_hls):
				z[j] = w[j].dot(acts[j]) + b[j]
				acts[j+1] = sigmoid(z[j])
			z[-1] = w[-1].dot(acts[-2]) + b[-1]
			acts[-1] = softmax(z[-1])
			if np.argmax(acts[-1]) == test_labels[k]:
				corr_labels += 1
		acc = corr_labels / t_size

		print('Validation accuracy in epoch ' + str(i) + ' is ' + str(acc))

		# update learning rate
		if i % 100 == 0 and i > 0:
			lr *= 0.1
			print('\nLearning rate is now ' + str(lr) + '\n')


if __name__ == "__main__":
	digits = np.arange(0, 10)
	path = 'mnist'
	download_mnist(path=path)
	train_images, train_labels = read(train=True, digits=digits, path=path)
	test_images, test_labels = read(train=False, digits=digits, path=path)
	neural_network(train_images, train_labels, test_images, test_labels, digits)