### Word2Vec skipgram


import tensorflow as tf
from process_data import process_data
from utils.data_process import make_dir

VOCAB_SIZE = 50000
EMBED_SIZE = 128
BATCH_SIZE = 128
SKIP_WINDOW = 1 # context window
NUM_SAMPLED = 64 # negative samples to sample
NUM_TRAIN_STEPS = 2000
LEARNING_RATE = 0.1
SKIP_STEP = 2000 # for loss print


def word2vec(batch_gen):
    # Defining placeholders for input (center_words) and output (target_words)
    # which are scalars. The input will be the words indices
    center_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    target_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

    # Weight matrix. Each row is the representation vector of one word and the
    # matrix is initialized to a value from a random distribution.
    intialize_embed = tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0)
    embed_matrix = tf.Variable(intialize_embed)

    # To avoid unnecessary matrix computation multiplying by the many zeros of a
    # one-hot vector 'tf.nn.embedding_lookup' is used to get the vector
    # representation (embedding) of the input center words
    embed = tf.nn.embedding_lookup(embed_matrix, center_words)

    # Hidden layer weights and biases for NCE loss
    nce_weigt = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE],
                                                    stddev=1.0 / EMBED_SIZE ** 0.5))
    nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]))

    # loss
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weigt,
                                         biases=nce_bias,
                                         labels=target_words,
                                         inputs=embed,
                                         num_sampled=NUM_SAMPLED,
                                         num_classes=VOCAB_SIZE))

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)


    # train
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        average_loss = 0.0
        for index in range(NUM_TRAIN_STEPS):
            batch = batch_gen.next()
            loss_batch, _ =  sess.run([loss, optimizer],
                                        feed_dict={center_words: batch[0],
                                                   target_words: batch[1]})
            average_loss += loss_batch
            if (index + 1) % 2000 == 0:
                print("Average loss at step {} {:5.1f}".format(index + 1,
                                                    average_loss / (index+ 1)))

def main():
    batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    word2vec(batch_gen)

if __name__ == '__main__':
    main()