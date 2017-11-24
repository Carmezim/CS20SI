### Word2Vec skipgram

import os

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from process_data import process_data
from utils import make_dir

VOCAB_SIZE = 50000
EMBED_SIZE = 128
BATCH_SIZE = 128
SKIP_WINDOW = 1 # context window
NUM_SAMPLED = 64 # negative samples to sample
NUM_TRAIN_STEPS = 200000
LEARNING_RATE = 1.0
WEIGHTS_FLD = "processed/"
SKIP_STEP = 2000 # for loss print

class SkipGram:
    def __init__(self, vocab_size, embed_size, batch_size,
                 num_sampled, learning_rate):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, dtype=tf.int32,
                                       trainable=False,
                                       name="global_step")


    # Defining placeholders for input (center_words) and output
    # (target_words) which are scalars. The input will be the words indices
    def _create_placeholders(self):
        with tf.name_scope("data"):
            self.center_words = tf.placeholder(tf.int32,
                                               shape=[self.batch_size],
                                               name="center_words")
            self.target_words = tf.placeholder(tf.int32,
                                               shape=[self.batch_size, 1],
                                               name="target_words")


    # Weight matrix. Each row is the representation vector of one word and
    # the matrix is initialized to a value from a random distribution.
    def _create_embedding(self):
        with tf.name_scope("embed"):
            initialize_embed = tf.random_uniform([self.vocab_size,
                                                self.embed_size], -1.0, 1.0)
            self.embed_matrix = tf.Variable(initialize_embed,
                                            name="embed_matrix")

    # To avoid unnecessary matrix computation multiplying by the many zeros of a
    # one-hot vector "tf.nn.embedding_lookup" is used to get the vector
    # representation (embedding) of the input center words
    def _create_loss(self):
        with tf.name_scope("loss"):
            embed = tf.nn.embedding_lookup(self.embed_matrix,
                                           self.center_words,
                                           name="embed")
            # Hidden layer weights and biases for NCE loss
            nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size,
                                                         self.embed_size],
                                        stddev=1.0 / self.embed_size ** 0.5),
                                        name="nce_weight")
            nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name="nce_bias")
            # loss
            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                                 biases=nce_bias,
                                                 labels=self.target_words,
                                                 inputs=embed,
                                                 num_sampled=self.num_sampled,
                                                 num_classes=self.vocab_size),
                                                name="loss")

    def _create_optimizer(self):
        # optimizer
        self.optimizer = tf.train.GradientDescentOptimizer(
                    self.learning_rate).minimize(self.loss,
                                                 global_step=self.global_step)

    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram loss", self.loss)
            # merge all summaries
            self.summary_op = tf.summary.merge_all()

    # assemble graph
    def build_graph(self):
        self._create_placeholders()
        self._create_embedding()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()


def train(model, batch_gen, num_steps, weights_fld):
    # defaults to save all variables
    saver = tf.train.Saver()

    initial_step = 0
    # create checkpoints directory
    make_dir("checkpoints")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # if checkpoint exists restore from it
        ckpt = tf.train.get_checkpoint_state(
                os.path.dirname("checkpoints/checkpoint"))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        # used to calculate late average loss in last SKIP_STEP
        total_loss = 0.0
        writer = tf.summary.FileWriter("improved_graph/learning_rate" + str(
            LEARNING_RATE), sess.graph)
        initial_step = model.global_step.eval()
        for index in range(initial_step, initial_step + num_steps):
            center_ws, targets = next(batch_gen)
            feed_dict = {model.center_words: center_ws,
                        model.target_words: targets}
            loss_batch, _, summary = sess.run([model.loss,
                                            model.optimizer,
                                            model.summary_op],
                                            feed_dict=feed_dict)
            writer.add_summary(summary, global_step=index)
            total_loss += loss_batch
            if (index + 1) % SKIP_STEP == 0:
                print("Average loss at step {} {:5.1f}".format(index,
                                                       total_loss / SKIP_STEP))
                total_loss = 0.0
                saver.save(sess, "checkpoints/skip-gram", index)

        # visualize embeddings
        final_embedding_matrix = sess.run(model.embed_matrix)
        embedding_var = tf.Variable(final_embedding_matrix[:1000],
                                    name="embedding")
        sess.run(embedding_var.initializer)
        config = projector.ProjectorConfig()
        summary_writer = tf.summary.FileWriter("processed")
        # add embedding to config flie
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        # link tensor to metadata file, e.g. the first 500 words of vocabulary
        embedding.metadata_path = "processed/vocab_1000.tsv"
        # saves a configuration file TensorBoard reads during startup
        projector.visualize_embeddings(summary_writer, config)
        saver_embed = tf.train.Saver([embedding_var])
        saver_embed.save(sess, "processed/model3.ckpt", 1)


def main():
    model = SkipGram(VOCAB_SIZE, EMBED_SIZE, BATCH_SIZE, NUM_SAMPLED, LEARNING_RATE)
    model.build_graph()
    batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    train(model, batch_gen, NUM_TRAIN_STEPS, WEIGHTS_FLD)

if __name__ == "__main__":
    main()