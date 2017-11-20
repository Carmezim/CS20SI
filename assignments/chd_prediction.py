import tensorflow as tf
import time

from utils import data_preprocess


FILE_PATH = "../data/heart.csv"
CATEGORICAL_COLUMN = "famhist"

# hyperparameters
n_features = 9
learning_rate = 0.01
epochs = 2000
n_hidden = 6
n_labels = 2

# data processing
data = data_preprocess.read_data(FILE_PATH)
data = data_preprocess.encode(data, CATEGORICAL_COLUMN)

columns = [ "sbp",
            "tobacco",
            "ldl",
            "adiposity",
            "typea",
            "obesity",
            "alcohol",
            "age"]

data = data_preprocess.scale_data(data, columns)

train_X, train_Y, test_X, test_Y, features, labels = data_preprocess.split_data(data)


def model():

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, 9], name="X")
    Y = tf.placeholder(tf.int32, [None], name="Y")
    one_hot_labels = tf.one_hot(Y, 2)

    w_h = tf.Variable(tf.truncated_normal([n_features, n_hidden], stddev=0.1))
    w_out = tf.Variable(tf.truncated_normal([n_hidden, n_labels], stddev=0.1))
    b_h = tf.Variable(tf.zeros(n_hidden)),
    b_out = tf.Variable(tf.zeros(n_labels))

    hidden_layer = tf.add(tf.matmul(X, w_h), b_h)
    hidden_layer = tf.nn.relu(hidden_layer)
    output = tf.add(tf.matmul(hidden_layer, w_out), b_out)

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=one_hot_labels)
    cost = tf.reduce_mean(loss)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    return X, Y, loss, cost, optimizer, output


def train(X, Y, cost, optimizer):
    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        print("Starting traning.")

        # tensorboard
        file_writer = tf.summary.FileWriter("./logs/1", sess.graph)
        start_time = time.time()

        for epoch in range(epochs):
            _, loss = sess.run([optimizer, cost], feed_dict={X: train_X, Y: train_Y})

            print("Epoch: {}".format(epoch))
            print("Training loss: {}".format(loss))

        print('Total time: {0} seconds'.format(time.time() - start_time))
        print("Optimization finished!")

        file_writer.close()


def test(X, Y, loss, output):
    with tf.Session() as sess:
        print("Testing data")

        sess.run(tf.global_variables_initializer())

        # tensorboard
        file_writer = tf.summary.FileWriter("./logs/2", sess.graph)
        start_time = time.time()

        test_loss, test_output = sess.run([loss, output], feed_dict={X: test_X, Y: test_Y})
        preds = tf.nn.softmax(test_output)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(tf.one_hot(test_Y, 2), 1))
        accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

        print("Model accuracy: {}".format(sess.run(accuracy)))



if __name__ == "__main__":
    X, Y, loss, cost, optimizer, output = model()
    train(X, Y, cost, optimizer)
    test(X, Y, loss, output)
