import re

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