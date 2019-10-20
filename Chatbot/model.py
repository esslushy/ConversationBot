import tensorflow as tf

def scaled_dot_product_attention(query, key, value, mask):
    """
      Args:
        query: the prompt being passed in
        key: used to calculate importance to the query
        value: the response from the model
        mask: used to zero out padding tokens
      
      This function makes sure valuable words are kept 
      as a focus and irrelavant words are flushed out
      toward 0. This returns the calculated attention
      weights for each query.
    """
    # Applies keys value to value
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # Scale matmul_qk by key
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # Add the mask zero out padding tokens.
    if mask is not None:
        # -1e9 is very close to negative infinity
        logits += (mask * -1e9)

    # Softmax normalized on the last axis
    attention_weights = tf.nn.softmax(logits, axis=-1)

    # Apply attention to output
    return tf.matmul(attention_weights, value)

class MultiHeadAttention(tf.keras.layers.Layer):
    """
      Args:
        query: the prompt being passed in
        key: used to calculate importance of the query
        value: the models response
        mask: used to zero out padding tokens
        d_model: the last section of data or the information about words
        num_heads: how many heads in the attention layer
        
      Multi-head attention consists of four parts:
        - Linear layers and split into heads.
        - Scaled dot-product attention.
        - Concatenation of heads.
        - Final linear layer.
      The scaled_dot_product_attention defined above 
      is applied to each head (broadcasted for efficiency). 
      An appropriate mask must be used in the attention step. 
      The attention output for each head is then concatenated 
      (using tf.transpose, and tf.reshape) and put through a 
      final Dense layer.

      Instead of one single attention head, query, key, and value
      are split into multiple heads because it allows the model to
      jointly attend to information at different positions from 
      different representational spaces. After the split each head 
      has a reduced dimensionality, so the total computation cost is 
      the same as a single head attention with full dimensionality
    """
    def __init__(self, d_model, num_heads, name='multi_head_attention'):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # Splits up inputs to appropiate inputs
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        # Calculate batch size
        batch_size = tf.shape(query)[0]

        # Linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # Split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # Scaled dot-product attention
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # Concatenation of heads
        concat_attention = tf.reshape(scaled_attention,
                                    (batch_size, -1, self.d_model))

        # Final linear layer
        outputs = self.dense(concat_attention)

        return outputs

# Masking
def create_padding_mask(x):
    """
      Args:
        x: the input vector to create the mask for

      Creates a padding mask to ignore the added padding tokens
    """
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # Note tf.newaxis is `None` in tensorflow
    return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(x):
    """
      Args:
        x: the input vector to create the mask for

      Creates a padding mask to ignore the future tokens in the sequence
      and for padding tokens. For example to predict the third word it 
      masks all words but the first and second.
    """
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)

# Encoding
class PositionalEncoding(tf.keras.layers.Layer):
    """
      Args:
        inputs: the inputs to apply the positional encoding to
        position: the position of the word within the sentence
        d_model: the last section of data or the information about words

      The positional encoding vector is added to the embedding vector. 
      Embeddings represent a token in a d-dimensional space where tokens 
      with similar meaning will be closer to each other. But the embeddings 
      do not encode the relative position of words in a sentence. So after
      adding the positional encoding, words will be closer to each other 
      based on the similarity of their meaning and their position in the 
      sentence, in the d-dimensional space. This gives the model some idea
      about the position of the word in the sentence.
    """
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        # Apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # Apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def encoder_layer(units, d_model, num_heads, dropout, name='encoder_layer'):
    """
      Args:
        units: units in the dense layer
        d_model: the last section of data or the information about words
        num_heads: number of heads for the multiheaded attention
        dropout: droupout rate for the dropout layers
        name: name of the layer

      Constructs the encoding section of the model through word inputs
      to a larger sentence understanding output.

      Each encoder layer consists of sublayers:
        - Multi-head attention (with padding mask)
        - 2 dense layers followed by dropout

      Each of these sublayers has a residual connection around it followed 
      by a layer normalization. Residual connections help in avoiding the 
      vanishing gradient problem in deep networks.
    """
    inputs = tf.keras.Input(shape=(None, d_model), name='inputs')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    attention = MultiHeadAttention(
        d_model, num_heads, name='attention')({
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': padding_mask
        })
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)# Note 1e-6 is the smallest allowed float32 value

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)
    # Returns build encoder layer to be used in the larger encoder
    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)

def encoder(vocab_size, num_layers, units, d_model, num_heads, dropout, name='encoder'):
    """
      Args:
        vocab_size: range of vocabulary used
        num_layers: number of encoding layers to use
        units: units in the dense layer
        d_model: the last section of data or the information about words
        num_heads: number of heads for the multiheaded attention
        dropout: rate of dropout
        name: name of the model

      Constructs the larger encoder model for use within the transformer.

      The Encoder consists of:
        - Input Embedding
        - Positional Encoding
        - num_layers encoder layers
    
      The input is put through an embedding which is summed with the positional encoding.
      The output of this summation is the input to the encoder layers. The output of the
      encoder is the input to the decoder.
    """
    # Build inputs and mask
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    # Word embedding system
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    # Dropout
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    # Add the number of encoding layers desired
    for i in range(num_layers):
        outputs = encoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name='encoder_layer_{}'.format(i),
        )([outputs, padding_mask])

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)

# Decoding 
def decoder_layer(units, d_model, num_heads, dropout, name='decoder_layer'):
    """
      Args:
        units: units in the dense layer
        d_model: the last section of data or the information about words
        num_heads: number of heads for the multiheaded attention
        dropout: droupout rate for the dropout layers
        name: name of the layer

      Each decoder layer consists of sublayers:

      Masked multi-head attention (with look ahead mask and padding mask)
        - Multi-head attention (with padding mask). value and key receive the encoder output as inputs. 
          query receives the output from the masked multi-head attention sublayer.
        - 2 dense layers followed by dropout

      Each of these sublayers has a residual connection around it followed 
      by a layer normalization. The output of each sublayer is LayerNorm(x + Sublayer(x)). 
      The normalization is done on the d_model (last) axis.

      As query receives the output from decoder's first attention block, and key receives the encoder output,
      the attention weights represent the importance given to the decoder's input based on the encoder's output.
      In other words, the decoder predicts the next word by looking at the encoder output and self-attending
      to its own output. 
    """
    inputs = tf.keras.Input(shape=(None, d_model), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name='look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    attention1 = MultiHeadAttention(
        d_model, num_heads, name='attention_1')(inputs={
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': look_ahead_mask
        })
    attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1 + inputs)

    attention2 = MultiHeadAttention(
        d_model, num_heads, name='attention_2')(inputs={
            'query': attention1,
            'key': enc_outputs,
            'value': enc_outputs,
            'mask': padding_mask
        })
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention2 + attention1)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask], outputs=outputs, name=name)

def decoder(vocab_size, num_layers, units, d_model, num_heads, dropout, name='decoder'):
    """
      Args:
        vocab_size: range of vocabulary used
        num_layers: number of decoding layers to use
        units: units in the dense layer
        d_model: the last section of data or the information about words
        num_heads: number of heads for the multiheaded attention
        dropout: rate of dropout
        name: name of the model

      Constructs the larger decoder model for use within the transformer.

      The Decoder consists of:
        - Output Embedding
        - Positional Encoding
        - N decoder layers
      
      The target is put through an embedding which is summed with the positional encoding. 
      The output of this summation is the input to the decoder layers. The output of the 
      decoder is the input to the final linear layer.
    """
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name='look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
    
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = decoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name='decoder_layer_{}'.format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask], outputs=outputs, name=name)

# Transformer
def transformer(vocab_size, num_layers, units, d_model, num_heads, dropout, name='transformer'):
    """
      Args:
        vocab_size: range of vocabulary used
        num_layers: number of decoding layers to use
        units: units in the dense layer
        d_model: the last section of data or the information about words
        num_heads: number of heads for the multiheaded attention
        dropout: rate of dropout
        name: name of the model

      Transformer consists of the encoder, decoder and a final linear layer. 
      The output of the decoder is the input to the linear layer and its output is returned.
    """
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

    # Maskings
    enc_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None), name='enc_padding_mask')(inputs)
    # Mask the future tokens for decoder inputs at the 1st attention block
    look_ahead_mask = tf.keras.layers.Lambda(create_look_ahead_mask, output_shape=(1, None, None), name='look_ahead_mask')(dec_inputs)
    # Mask the encoder outputs for the 2nd attention block
    dec_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None), name='dec_padding_mask')(inputs)
    # Encoder
    enc_outputs = encoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[inputs, enc_padding_mask])
    # Decoder
    dec_outputs = decoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    outputs = tf.keras.layers.Dense(units=vocab_size, name='outputs')(dec_outputs)

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)