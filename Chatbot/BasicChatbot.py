import tensorflow as tf
# Used to make sure randomness is constant
tf.random.set_seed(7)
import tensorflow_datasets as tfds
import re
import os
# Global settings and parameters
from settings import *
# Model
from model import transformer
# Preprocessing
from preprocessing import preprocess_sentence

# Make directories for files
os.mkdir(MODEL_LOCATION)
os.mkdir(TOKENIZER_LOCATION)
os.mkdir(TENSORBOARD_LOCATION)

# Download Dataset. New tool from keras
path_to_zip = tf.keras.utils.get_file('cornell_movie_dialogs.zip', 
                                    origin='http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip', extract=True)
# Get directory names
path_to_dataset = os.path.join(os.path.dirname(path_to_zip), "cornell movie-dialogs corpus")
# Get the lines and the conversation hooks
path_to_movie_lines = os.path.join(path_to_dataset, 'movie_lines.txt')
path_to_movie_conversations = os.path.join(path_to_dataset, 'movie_conversations.txt')

# Data Loading Function
def load_data():
    # Dictionary for line id to the line used
    id_to_line =  {}
    # Read movie lines file
    with open(path_to_movie_lines, errors='ignore') as f:
        lines = f.readlines()
    # Break up each line into core components
    for line in lines:
        # Breaks line into an array containing the id and line (middle values irrelevant)
        parts = line.replace('\n', '')
        parts = parts.split(' +++$+++ ')
        # Stores the id and line in our dictionary
        id_to_line[parts[0]] = parts[4]
    # Build Input Output pairs
    inputs, outputs = [], []
    # Read the conversation file
    with open(path_to_movie_conversations, errors='ignore') as f:
        lines = f.readlines()
    # For each line pull out the array of lines
    for line in lines:
        # Pull out each major section from the conversation.
        parts = line.replace('\n', '')
        parts = parts.split(' +++$+++ ')
        # Convert string of list of line ids to an actual list of the line ids.
        # Takes the stringified array, removes the brackets, splits each line id, then extracts each line id without the wrapping quotes
        conversation = [l[1:-1] for l in parts[3][1:-1].split(', ')]
        for i in range(len(conversation) -1):
            # Adds each input and output to the arrays instantiated earlier
            inputs.append(preprocess_sentence(id_to_line[conversation[i]]))
            outputs.append(preprocess_sentence(id_to_line[conversation[i+1]]))
    return inputs, outputs

prompts, responses = load_data()

# Build tokenizer using tfds for both prompts and responses. This turns words into positive integers.
tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    prompts + responses, target_vocab_size=2**13)

# Define start and end token to indicate the start and end of a sentence
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

# Vocabulary size plus start and end token
VOCAB_SIZE = tokenizer.vocab_size + 2

# Tokenize, filter and pad sentences
def tokenize_and_filter(inputs, outputs):
  tokenized_inputs, tokenized_outputs = [], []
  
  for (sentence1, sentence2) in zip(inputs, outputs):
    # Tokenize sentences
    sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
    sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
    # Check tokenized sentence max length. If it exceeds remove it.
    if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
      tokenized_inputs.append(sentence1)
      tokenized_outputs.append(sentence2)
  
  # Pad tokenized sentences so that they are all of consistent length.
  tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
  tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_outputs, maxlen=MAX_LENGTH, padding='post')
  
  return tokenized_inputs, tokenized_outputs

prompts, responses = tokenize_and_filter(prompts, responses)

# Build the dataset
# Decoder inputs use the previous target as part of the input
# Remove START_TOKEN from targets
dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': prompts,
        'dec_inputs': responses[:, :-1]
    },
    {
        'outputs': responses[:, 1:]
    },
))

# Apply operations to speed up dataset loading, shuffle the dataset, batch samples.
dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
# Dataset inputs ((None, 40), (None, 39)) and outputs (None, 39)

# Build model
model = transformer(vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, units=UNITS, d_model=D_MODEL, num_heads=NUM_HEADS, dropout=DROPOUT)

# Loss function
def loss_function(y_true, y_pred):
    # Clip y_true within max length
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
    # Sparse categorical loss
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(y_true, y_pred)
    # Mask the loss function to only care about non-zero inputs
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)

# Learning rate schedule
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
      Args:
        d_model: the last section of data or the information about words
        warmup_steps: learning rate acceleration time period before drop off

      Calculates learning rate according to the paper by this formula:
        learning_rate=d_model^-0.5 * min(step_num^-0.5, step_num*warmup_steps^-1.5)
    """
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
    # Override get_config
    def get_config(self):
        return {
            'd_model' : self.d_model,
            'warmup_steps' : self.warmup_steps
        }

# Compile model
learning_rate = CustomSchedule(D_MODEL)
# Adam optimizer with some small changes. Very tiny epsilon and much higher beta_1
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
# Modified accuracy calculation to work with model
sparse_categorical_accuracy = tf.metrics.SparseCategoricalAccuracy()
def accuracy(y_true, y_pred):
    # ensure labels have shape (batch_size, MAX_LENGTH - 1)
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
    accuracy = sparse_categorical_accuracy(y_true, y_pred)
    return accuracy

model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy],
            callbacks=[tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOCATION, update_freq='batch', profile_batch=0)])

# Train model
model.fit(dataset, epochs=EPOCHS)

# Save model
tf.saved_model.save(model, MODEL_LOCATION)
tokenizer.save_to_file(TOKENIZER_LOCATION)