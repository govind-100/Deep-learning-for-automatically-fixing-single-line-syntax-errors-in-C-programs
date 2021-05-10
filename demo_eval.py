import ast
import os
import sys

import numpy as np
import dill as dill
import pandas as pd
import tensorflow as tf
from Decoder import Decoder
from Encoder import Encoder
import operator


if (len(sys.argv)) != 3:
    exit(1)

input_file_name = sys.argv[1]
output_file_name = sys.argv[2]


def padding(l, pad, width):
    l.extend([pad] * (width - len(l)))
    return l


def evaluate(sentence):
    sourceTensor = np.zeros(shape=(1, 100))

    for i in range(1):
        j = 0
        for word in sentence:
            if word in mostFreqTokens:
                sourceTensor[i][j] = voc.to_index(word)
            else:
                sourceTensor[i][j] = voc.OOV_token
            j += 1

    inputs = tf.convert_to_tensor(sourceTensor)

    result = []

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([voc.word2index['SOS']], 0)

    for t in range(targetTensor.shape[1]):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1,))

        predicted_id = tf.argmax(predictions[0]).numpy()
        if voc.index2word[predicted_id] != 'EOS':
            result.append(voc.index2word[predicted_id])

        if voc.index2word[predicted_id] == 'EOS':
            return result, sentence

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence


def translate(sentence):
    result, sentence = evaluate(sentence)
    source = []
    for word in sentence:
        if word == 'SOS' or word == 'EOS' or word == 'PAD':
            continue
        else:
            source.append(word)

    return result



data = pd.read_csv(input_file_name
                   )

sourceLineTokens = data['sourceLineTokens']
targetLineTokens = data['targetLineTokens']

sourceLineTokens = [["SOS"] + ast.literal_eval(token)[:98] + ["EOS"] for token in sourceLineTokens]
sourceLineTokens = [padding(token, "PAD", 100) for token in sourceLineTokens]

with open(f"./mostFrequentK.pkl", "rb") as f:
    voc = dill.load(f)

# print(voc.word2count)

# for i in range(len(sourceLineTokens)):
#     voc.createVoc(sourceLineTokens[i])

mostFreqTokens = dict((sorted(voc.word2count.items(), key=operator.itemgetter(1), reverse=True))[:500])
sourceTensor = np.zeros(shape=(len(sourceLineTokens), 100))
targetTensor = np.zeros(shape=(len(targetLineTokens), 100))

BUFFER_SIZE = len(sourceTensor)
BATCH_SIZE = 64
steps_per_epoch = len(sourceTensor)
embedding_dim = 150
units = 64
voc_input_size = len(voc.word2index) + 1
voc_target_size = len(voc.word2index) + 1
dataset = tf.data.Dataset.from_tensor_slices((sourceTensor, targetTensor)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
example_input_batch, example_target_batch = next(iter(dataset))

encoder = Encoder(voc_input_size, embedding_dim, units, BATCH_SIZE)

decoder = Decoder(voc_target_size, embedding_dim, units, BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

checkpoint_dir = 'model_checkpoint'
checkpoint_prefix = os.path.join(checkpoint_dir, "checkpoint")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


output = []

for i in range(len(sourceLineTokens)):
    output.append(translate(sourceLineTokens[i]))


data["fixedTokens"] = output


data.to_csv(output_file_name)