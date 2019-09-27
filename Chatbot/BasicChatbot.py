import tensorflow as tf
# Used to make sure randomness is constant
tf.random.set_seed(7)
import tensorflow_datasets as tfds
import re
import os

# Download Dataset. New tool from keras
path_to_zip = tf.keras.utils.get_file('cornell_movie_dialogs.zip', 
                                    origin='http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip', extract=True)
# Get directory names
path_to_dataset = os.path.join(os.path.dirname(path_to_zip), "cornell movie-dialogs corpus")
# Get the lines and the conversation hooks
path_to_movie_lines = os.path.join(path_to_dataset, 'movie_lines.txt')
path_to_movie_conversations = os.path.join(path_to_dataset, 'movie_conversations.txt')

# Preprocessing Function
def preprocess_sentence(sentence):
    # Makes the sentence lowercase and removes trailing whitespace
    sentence = sentence.lower().strip()
    # Places a space between words and punctuation for clarity
    sentence = re.sub(r'([?.!,;])', r' \1', sentence)
    # Turns all whitespace into a whitespace of length one for easier use later
    sentence = re.sub(r'[" "]+', ' ', sentence)
    # Replace everything with a space that doesn't match use punctuation, words, or numbers
    sentence = re.sub(r'[^a-zA-z?.!,; ]+', '', sentence)
    # Strip any added trailing white space after this
    sentence = sentence.strip()
    return sentence

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

print(prompts[20], '\n',  responses[20])