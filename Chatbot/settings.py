# Maximum sentence length
MAX_LENGTH = 40
# Batch size
BATCH_SIZE = 64
# Shuffle Buffer
BUFFER = 20000
# Number of epochs to train for
EPOCHS = 200
# Hyper-parameters. Made according to the paper: https://arxiv.org/abs/1706.03762
NUM_LAYERS = 2
D_MODEL = 256
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1
# Logging information
TENSORBOARD_LOCATION = 'info/run1'
# Save locations for model and tokenizer
MODEL_LOCATION = 'model/run1'
TOKENIZER_LOCATION = 'tokenizer/run1'