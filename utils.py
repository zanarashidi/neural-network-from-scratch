import numpy as np 
import os
import gzip
import urllib.request
import struct


def download_mnist(path='mnist'):
    os.makedirs(path, exist_ok=True)

    base_url = 'http://yann.lecun.com/exdb/mnist/'
    files = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
             't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']

    for file in files:
        url = f'{base_url}{file}.gz'
        local_path = os.path.join(path, file)

        # Download the file if not already present
        if not os.path.exists(local_path):
            print(f'Downloading {file}...')
            urllib.request.urlretrieve(url, local_path + '.gz')

            # Extract the gz file
            with gzip.open(local_path + '.gz', 'rb') as f_in, open(local_path, 'wb') as f_out:
                f_out.write(f_in.read())

            print(f'{file} downloaded successfully.')
        else:
            print(f'{file} already exists.')
    print('\n')


def read(train=True, digits=np.arange(10), path='.'):
	if train:
		print('Reading training data ...\n')
		image_file = os.path.join(path, 'train-images-idx3-ubyte')
		label_file = os.path.join(path, 'train-labels-idx1-ubyte')
		total_size = 60000
	else:
		print('Reading testing data ...\n')
		image_file = os.path.join(path, 't10k-images-idx3-ubyte')
		label_file = os.path.join(path, 't10k-labels-idx1-ubyte')
		total_size = 10000

	with open(label_file, 'rb') as lbf:
		magic, num = struct.unpack('>II', lbf.read(8))
		label = np.fromfile(lbf, dtype=np.int8)

	with open(image_file, 'rb') as imf:
		magic, num, rows, cols = struct.unpack('>IIII', imf.read(16))
		image = np.fromfile(imf, dtype=np.uint8).reshape(len(label), rows, cols)

	idx = [i for i in range(total_size) if label[i] in digits]
	size = int(len(idx))
	images = np.zeros((size, rows, cols), dtype=np.uint8)
	labels = np.zeros((size, 1), dtype=np.int8)

	for i in range(size):
		images[i] = image[idx[i]]
		labels[i] = label[idx[i]]

	labels = [label_i[0] for label_i in labels]
	return images, labels


if __name__ == "__main__":
	images, _ = read(train=True, digits=np.array([5]), path='./mnist')
	show(images[0])




