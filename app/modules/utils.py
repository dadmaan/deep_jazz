import os
import tensorflow as tf
import numpy as np
import json
import random

class TextDatasetLoader():
    """ ABC music notation dataset loader"""
    
    def __init__(self,
                 sequence_length,
                 batch_size,
                 buffer_size,
                 text_dir='dataset/abc/tunes.txt',
                 output_dir='dataset/abc/vocab_tunes.json',
                 dataset_size='full'):
        
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.text_dir = text_dir
        self.output_dir = output_dir
        self.vocab_length = 0
        self.char2idx = {}
        self.idx2char = []
        self.dataset_length = 0
        self.ds_size = dataset_size
    
    def unique_chars(self, dataset):
        """ Determines the unique character in the dataset """
        return sorted(list(set(dataset)))

    def unique_words(self, dataset):
        """ Define the unique words inside the text file."""
        words = dataset.split()
        words = [word.lower() for word in words if len(word)>1 and word.isalpha()]
        return sorted(list(set(words)))
    
    def get_split(self, dataset, split):
        """ returns the trainig/validation/testing examples """
        if split == 'train':
            length = int(self.dataset_length * 0.8)
            return dataset[:length]
        if split == 'valid':
            begin = int(self.dataset_length * 0.8)
            end = int(self.dataset_length *0.9)
            return dataset[begin:end]
        if split == 'test':
            begin = int(self.dataset_length *0.9)
            end = int(self.dataset_length)
            return dataset[begin:end]
        
    def char2index(self, vocab):
        """ Map the vocabulary characters to corresponding index"""
        c2i = {ch : i for i,ch in enumerate(vocab)}
        return c2i

    def index2char(self, vocab):
        """ Map the indices to corresponding characters in vocabulary"""
        return np.array(vocab)

    def split_input_target(self, chunk):
        """ Split the dataset into chunks of input/target for each dataset.
            Simply duplicate the given sequence and shift it. 
        """
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    def load_vocab(self, path):
        """ Load the vocab file """
        with open(path) as f:
            vocab = json.load(f)
        return vocab
    
    def get_vocab_length(self):
        return self.vocab_length
    
    def get_char2index(self):
        return self.char2idx
    
    def get_index2char(self):
        return self.idx2char
    
    def get_stepsize(self):
        return int(self.dataset_length / (self.sequence_length * self.batch_size))
    
    def get_dataset_length(self):
        return self.dataset_length
    
    def get_subdataset(self, dataset, size):
        """ Reduces size of the dataset """
        if size == 'small':
            length = int(len(dataset)//6)
            dataset = dataset[:length]
            self.dataset_length = len(dataset)
            return dataset
        
        if size == 'medium':
            length = int(len(dataset)//4)
            dataset = dataset[:length]
            self.dataset_length = len(dataset)
            return dataset
        
        if size == 'large':
            length = int(len(dataset)//2)
            dataset = dataset[:length]
            self.dataset_length = len(dataset)
            return dataset
        
    def load_dataset(self, split=None):
        """ Get the music text file in ABC notation, 
            prepare the training examples/targets with tensorflow tf.dataset for RNN model.
            Here, we generate each batch that contains sequence of examples with the same length
            except one character shifted to the right

            Return:
            Tensorflow dataset
        """

        # Fetch the text file
        text = open(self.text_dir, 'r').read()
        self.dataset_length = len(text)

        # Get the unique characters
        vocab = self.unique_chars(text)
#         vocab = self.unique_words(text)
        self.vocab_length = len(vocab)

        # Save the unique characters
        with open(self.output_dir, 'w') as f:
            json.dump(vocab, f)

        # Map unique characters to indices
        self.char2idx = self.char2index(vocab)
        # Map indices to unique characters
        self.idx2char = self.index2char(vocab)

        # Map the dataset characters to int value
        text_as_int = np.array([self.char2idx[c] for c in text])
        
        # Define the size of dataset
        if not(self.ds_size == 'full'):
            text_as_int = self.get_subdataset(text_as_int, self.ds_size)
        
        # Get the right split from the dataset
        if not(split == None):
            text_as_int = self.get_split(text_as_int, split)
        
        # Preparing the tensorflow pipeline
        char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

        # Create the sequence of examples; each sequence shifted one character
        sequences = char_dataset.batch(self.sequence_length + 1, drop_remainder=True)

        # Parse each batch into a dataset of [input, target] example pairs
        ds = sequences.map(self.split_input_target)

        # Shuffle the dataset 
        ds = ds.shuffle(self.buffer_size)

        # Create the batches
        ds = ds.batch(self.batch_size, drop_remainder=True)

        return ds
    
def save_text(data, output_dir):
    
    with open(output_dir, 'w') as f:
        f.write(music)
    print('File saved!')
    
def load_text(abc_dir):
    with open(abc_dir) as f:
        x = f.read().decode(encoding='utf-8')
    print('File loaded!')
    return x

def composer(model, start_string, length, temperature, encoder, decoder):
    counter = 0
    
    # Number of characters to generate
    num_generate = length
    
    # Converting our start string to numbers (vectorizing)
    input_eval = [encoder[s] for s in start_string]

    input_eval = tf.expand_dims(input_eval, 0)
    
    # Empty string to store our results
    music_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temp = temperature

    # Here batch size == 1
    model.reset_states()
    while True:
        predictions = model(input_eval)
        
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temp
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        
        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        
        input_eval = tf.expand_dims([predicted_id], 0)

        music_generated.append(decoder[predicted_id])
        
        counter += 1
        if counter > num_generate:
            break
#         if idx2char[predicted_id] == '|':
#             flag_end = True
            
#         if counter > num_generate and idx2char[predicted_id] == ']':
#             if flag_end == True:
#                 break        
#             flag_end = False
            
#         if counter > 3000 and idx2char[predicted_id] == ']':
#             break 
        
    return (start_string + ''.join(music_generated))