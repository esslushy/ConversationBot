import tensorflow as tf

from settings import *
from preprocessing import preprocess_sentence

# Load tokenizer
tokenizer = tfds.features.text.SubwordTextEncoder()
tokenizer.load_from_file(TOKENIZER_LOCATION)

# Remake tokens
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

# Load model
model = tf.saved_model.load(MODEL_LOCATION)

def evaluate(sentence):
    sentence = preprocess_sentence(sentence)

    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

    output = tf.expand_dims(START_TOKEN, 0)

    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        # concatenated the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def predict(sentence):
    prediction = evaluate(sentence)

    predicted_sentence = tokenizer.decode([i for i in prediction if i < tokenizer.vocab_size])

    print('Output: {}'.format(predicted_sentence))

    return predicted_sentence

prompt = input('Hello, my name is Turry. I am a chatbot. Currently I am stupid, but later I will be smarter. Ask me anything: \n')
while(prompt != 'quit'):
    predict(prompt)
    prompt = input()