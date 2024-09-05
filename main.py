import keras
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import string
import os
from PIL import Image
import glob
from pickle import dump, load
from time import time
import cv2

from keras.api.preprocessing import sequence
from keras.api.models import Sequential
from keras.api.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Activation, Flatten, Reshape, \
    concatenate, Dropout, BatchNormalization
from keras.src.optimizers import Adam, RMSprop
from keras.src.layers import Bidirectional
from keras.src.layers.merging import add
from keras.src.layers.merging.add import Add
from keras.src.applications.inception_v3 import InceptionV3
from keras.api.preprocessing import image
from keras.api.models import Model
from keras.api import Input, layers
from keras import optimizers
from keras.src.applications.inception_v3 import preprocess_input
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras.api.preprocessing.sequence import pad_sequences
from keras.api.utils import to_categorical



# read file captions
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


filename = 'Flickr8k_text/Flickr8k.token.txt'

doc = load_doc(filename)


# Store caption as key-value
def load_descriptions(doc):
    mapping = dict()
    # process lines
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        if len(line) < 2:
            continue
        # take the first token as the image_id, the rest as the description
        image_id, image_desc = tokens[0], tokens[1:]
        # extract filename from image_id
        image_id = image_id.split('.')[0]
        # convert description tokens back to string
        image_desc = ' '.join(image_desc)
        # create the list if needed
        if image_id not in mapping:
            mapping[image_id] = list()
        # store description
        mapping[image_id].append(image_desc)
    return mapping


descriptions = load_descriptions(doc)


# print(f'Loaded: {len(descriptions)}')
#
# print(descriptions['1000268201_693b08cb0e'])

# Preprocessing text
def clean_descriptions(descriptions):
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # tokenize
            desc = desc.split()
            # convert to lower case
            desc = [word.lower() for word in desc]
            # remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            # remove hanging 's' and 'a'
            desc = [word for word in desc if len(word) > 1]
            # remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha()]
            # store as string
            desc_list[i] = ' '.join(desc)


clean_descriptions(descriptions=descriptions)


# print(descriptions['1000268201_693b08cb0e'])

# save descriptions to file
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
        data = '\n'.join(lines)
        file = open(filename, 'w')
        file.write(data)
        file.close()


# save_descriptions(descriptions, 'descriptions.txt')

# get the image ids corresponding to train, dev, test data
def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    # process line by line
    for line in doc.split('\n'):
        # skip empty line
        if len(line) < 1:
            continue
        # get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)


# load training dataset
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'

train = load_set(filename)
# print(f'dataset: {len(train)}')

# folder containing images
images = 'Flickr8k_Dataset/Flicker8k_Dataset/'
# get image with .jpg extension
img = glob.glob(images + '*.jpg')

# file containing the image id for training
train_images_file = 'Flickr8k_text/Flickr_8k.trainImages.txt'
# read the train image names in a set
train_images = set(open(train_images_file, 'r').read().strip().split('\n'))

# create a list of all the training images with their full path names
train_img = []

for i in img:
    if i[len(images):] in train_images:  # Check if the image belongs to training set
        train_img.append(i)

# file containing the image id for testing
test_images_file = 'Flickr8k_text/Flickr_8k.testImages.txt'
test_images = set(open(test_images_file, 'r').read().strip().split('\n'))

# create a list of all the test images with their full path names
test_img = []

for i in img:
    if i[len(images):] in test_images:
        test_img.append(i)


# add 'startseq', 'endseq' for sequence
def load_clean_descriptions(filename, dataset):
    # load doc
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        # split id from descriptions
        image_id, image_desc = tokens[0], tokens[1:]
        # skip images not in the set:
        if image_id in dataset:
            # create list
            if image_id not in descriptions:
                descriptions[image_id] = list()
            # wrap description in tokens
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            # store
            descriptions[image_id].append(desc)
    return descriptions


# train descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)


# print(len(train_descriptions))

# load image, resize ve kich thuoc ma Inception v3 yeu cau
def preprocess(image_path):
    # convert all the image to size 299x299 as expected by the inception v3 model
    img = image.load_img(image_path, target_size=(299, 299))
    # convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # add one more dimensions
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception v3 module
    x = preprocess_input(x)
    return x


# Load the Inception v3 model
model = InceptionV3(weights='imagenet')

# create new model, remove last layer from inception v3
model_new = Model(model.input, model.layers[-2].output)


# Image embedding thanh vector(2048, )
def encode(image):
    image = preprocess(image)
    fea_vec = model_new.predict(image)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec


# call encode function with images in training set
# start = time()
# encoding_train = {}
# for img in train_img:
#     encoding_train[img[len(images):]] = encode(img)
# print('time taken in seconds = ', time() - start)
#
# #Save image embedding
# with open('Pickle/encoded_train_images.pkl', 'wb') as encoded_pickle:
#     dump(encoding_train, encoded_pickle)
#
# #Encode test image
# start = time()
# encoding_test = {}
# for img in test_img:
#     encoding_test[img[len(images):]] = encode(img)
# print('time take in seconds = ', time() - start)
#
# #save the bottleneck test features to disk
# with open('Pickle/encoded_test_images.pkl', 'wb') as encoded_pickle:
#     dump(encoding_test, encoded_pickle)

train_features = load(open('Pickle/encoded_train_images.pkl', 'rb'))
# print(len(train_features))


# create list of training caption
all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)

# print(len(all_train_captions))

# get only word xuat hien >= 10 lan
word_count_threshold = 10
word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
# print(f'{len(word_counts)} -> {len(vocab)}')

ixtoword = {}
wordtoix = {}

ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

vocab_size = len(ixtoword) + 1


# convert a dict of clean descriptions to a list of descriptions
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc


# caculate the length of the description with the most words
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)


# determine the maximum sequence length
max_length = max_length(train_descriptions)


# data generator for train in batches
def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch):
    x1, x2, y = list(), list(), list()
    n = 0
    # loop for ever over image
    while 1:
        for key, desc_list in descriptions.items():
            n += 1
            # retrieve the photo feature
            photo = photos[key + '.jpg']
            for desc in desc_list:
                # encode the sequence
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                # split one sequence into multiple x,y pairs
                for i in range(1, len(seq)):
                    # split into input nad output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    x1.append(photo)
                    x2.append(in_seq)
                    y.append(out_seq)
                # yield the batch data
                if n == num_photos_per_batch:
                    yield (array(x1), array(x2)), array(y)
                    x1, x2, y = list(), list(), list()
                    n = 0

#load glove model
glove_dir = 'glove.6B'
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.200d.txt'), encoding='utf-8')

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
# print(f'found {len(embeddings_index)} word vector')
#
# print(embeddings_index['the'])


embedding_dim = 200
#get 200-dim dense vector for each of the 10000 words in out vocabulary
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in wordtoix.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        #word not found in the embedding index will be all zeros
        embedding_matrix[i] = embedding_vector

# print(embedding_matrix.shape)

#Model
# inputs1 = Input(shape=(2048,))
# fe1 = Dropout(0.5)(inputs1)
# fe2 = Dense(256, activation='relu')(fe1)
# inputs2 = Input(shape=(max_length,))
# se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
# se2 = Dropout(0.5)(se1)
# se3 = LSTM(256)(se2)
# decoder1 = Add()([fe2,se3])
# decoder2 = Dense(256, activation='relu')(decoder1)
# outputs = Dense(vocab_size, activation='softmax')(decoder2)
# model = Model(inputs=[inputs1, inputs2], outputs=outputs)
#
# model.summary()
#
# #layer 2 dung Glove model nen set weight thang va khong can train
# model.layers[2].set_weights([embedding_matrix])
# model.layers[2].trainable = False
# model.compile(loss='categorical_crossentropy', optimizer='adam')
#
# model.optimizer.lr = 0.0001
# epochs = 10
# number_pics_per_batch = 6
# steps = len(train_descriptions)//number_pics_per_batch
#
# for i in range(epochs):
#     generator = data_generator(train_descriptions, train_features, wordtoix, max_length, number_pics_per_batch)
#     model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
#
# model.save('image_captioning.keras')



images = 'Flickr8k_Dataset/Flicker8k_Dataset/'

with open('Pickle/encoded_test_images.pkl', 'rb') as encoded_pickle:
    encoding_test = load(encoded_pickle)

saved_model = keras.models.load_model('image_captioning.keras')

#Với mỗi ảnh mới khi test, ta sẽ bắt đầu sequence với 'startseq' rồi sau đó cho vào model để
#predict từ tiếp theo. Ta thêm word vừa predicted vào sequence và tiếp tục cho đến khi gặp 'endseq'
#là end hoặc cho đến khi sequence dài 34 word

def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = saved_model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

z = 95
pic = list(encoding_test.keys())[z]
image = encoding_test[pic].reshape((1,2048))
x=plt.imread(images+pic)
plt.imshow(x)
plt.title(greedySearch(image), fontsize=12)
plt.show()


