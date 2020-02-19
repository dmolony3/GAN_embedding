import tensorflow as tf
import os
import numpy as np
from PIL import Image

class DataReader():
    """Reads images from text file"""
    def __init__(self, directory, batch_size):
        self.directory = directory
        self.batch_size = batch_size

    def read_files(self, file_path):
        file_list = []
        f = open(file_path, 'r')
        for line in f:
            if line:
                file_list.append(os.path.join(self.directory, line.strip()))
        f.close()
        return file_list

    def decode_image(self, image):
        image = tf.io.read_file(image)
        image = tf.io.decode_png(image, channels=1)
        return image

    def read_batch(self, file_path):
        file_list = self.read_files(file_path)
        data = tf.data.Dataset.from_tensor_slices((file_list))
        data = data.map(self.decode_image)
        data = data.batch(batch_size=self.batch_size, drop_remainder=False)
        return data


directory = '/home/microway/Documents/IVUS/Segmentation2.0/Embedding_GAN/Images_20MHz'
data_file = '/home/microway/Documents/IVUS/Segmentation2.0/Embedding_GAN/generated.txt'
num_image_rows = 60 # must be divisble by batch_size
embedding_dim = 64
image_dim = 128
use_PCA = 1 # flag indicating whether PCA should be applied to first reduced the dimensions to 50

sprite_file = 'sprite_image' + '_' + str(embedding_dim) + 'x' + str(embedding_dim) +'.jpg'
vecs_file = 'vecs' + '_' + str(embedding_dim) + 'x' + str(embedding_dim) +'.tsv'
meta_file = 'metadata' + '_' + str(embedding_dim) + 'x' + str(embedding_dim) +'.tsv'

batch_size = 1
num_images = num_image_rows**2

data = DataReader(directory, batch_size)
data_iterator = iter(data.read_batch(data_file))

image_list = []
sprite_list = []

for i in range(num_images):
    batch = next(data_iterator)
    batch_downsampled = tf.image.resize(batch, [embedding_dim, embedding_dim])
    batch_vectorized = tf.reshape(batch_downsampled, shape=[batch_size, embedding_dim*embedding_dim])
    batch_vectorized = tf.squeeze(batch_vectorized, 0)
    image_list.append(batch_vectorized)
    sprite_list.append(tf.squeeze(tf.squeeze(tf.image.resize(batch, [image_dim, image_dim]), 0), -1))

if use_PCA == 1:
    images = tf.stack(image_list, 0)
    images = tf.cast(images, tf.float32)
    images = tf.transpose(images, [1, 0]) # [d x n]

    d = images.shape[0]
    n = images.shape[1]
    data_mean = tf.reduce_mean(images, axis=1) # [d]

    # compute the covariance matrix
    cov = 1/(d-1)*tf.matmul((images - tf.expand_dims(data_mean, 1)), 
                        tf.transpose(images - tf.expand_dims(data_mean, 1), [1, 0]))
    eigenvalues, eigenvectors = tf.linalg.eigh(cov)
    
    #sort eigenvectors by decreasing values
    tf.argsort(eigenvalues, direction='DESCENDING')
    eig_pairs = [(tf.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(d)]
    eig_pairs.sort(key = lambda x: x[0], reverse=True)

    # reduce to 50d space i.e. construct d x k matrix
    matrix_W = tf.stack([tf.reshape(eig_pairs[i][1], [d, 1]) for i in range(50)],axis=1)
    matrix_W = tf.squeeze(matrix_W, -1)
    # transform original data to subspace
    transformed = tf.matmul(tf.transpose(matrix_W, [1, 0]), images) # (d x k) x (k x n)

if use_PCA == 0:
    images = tf.stack(image_list, 0)
elif use_PCA == 1:
    images = tf.transpose(transformed, [1, 0])
dim = images.shape[1]

f1 = open(vecs_file, 'a', encoding='utf-8')
f2 = open(meta_file, 'a', encoding='utf-8')
for i in range(num_images):
    if i % 100 == 0:
        print(i)
    f1.write('\t'.join([str(images[i, :].numpy()[j]) for j in range(dim)]) + "\n")
    if i < (num_image_rows**2)//2:
        f2.write('{}\n'.format('fake'))
    else:
        f2.write('{}\n'.format('real'))
f1.close()
f2.close()

# generate the sprite image
sprite_image = np.zeros((image_dim*num_image_rows, image_dim*num_image_rows))
idx = 0
for i in range(num_image_rows):
    for j in range(num_image_rows):
        sprite_image[i*image_dim:(i+1)*image_dim, j*image_dim:(j+1)*image_dim] = sprite_list[idx]
        idx += 1
im = Image.fromarray(sprite_image.astype(np.uint8))
im.save(sprite_file)

