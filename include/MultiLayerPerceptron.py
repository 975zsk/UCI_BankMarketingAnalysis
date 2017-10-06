import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, recall_score, precision_score, accuracy_score
import tempfile
import os

# A multilayer perceptron in tensorflow that probably needs further tuning
# to work well in data provided

def MultiLayerPerceptron(X , labels , X_test , y_test, verbose=1):

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    display_step = 100
    n_hidden_1 = 20
    n_hidden_2 = 10
    beta = 0.01
    lr = 0.001
    l2 = True
    training_epochs = 1000
    dropout = 0.7
    LOG_DIR = os.path.join(tempfile.gettempdir(), r'MLP_run')
    decay_factor = 0.95

    n_input = np.shape(X)[1]
    n_classes = 2

    # Graph input
    x = tf.placeholder("float", [None, n_input],name='x')
    y = tf.placeholder("float", [None, n_classes],name='labels')
    keep_prob = tf.placeholder("float",name='keep_prob')


    def multilayer_perceptron(x, weights, biases, keep_prob):

        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])             # 1st hidden layer
        layer_1 = tf.nn.relu(layer_1,name='layer1_relu')                        # RELU activation

        drop_out1 = tf.nn.dropout(layer_1, keep_prob)                           # Dropout for regularization

        layer_emb = tf.add(tf.matmul(drop_out1, weights['h2']), biases['b2'])     # 2nd hidden layer
        layer_2 = tf.nn.relu(layer_emb,name='layer2_relu')                        # RELU activation

        drop_out2 = tf.nn.dropout(layer_2, keep_prob)                            # Dropout for regularization

        out_layer = tf.add(tf.matmul(drop_out2, weights['out']), biases['out'],name='out_layer')
        tf.summary.histogram("out_layer", out_layer, collections=['Train'])

        return out_layer, layer_emb

    # Initialize and store weights & biases
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    # Construct model
    with tf.name_scope('Model'):
        pred, layer_emb = multilayer_perceptron(x, weights, biases,keep_prob)

    # Weighted loss for dealing with skewed class
    with tf.name_scope('loss'):
        cost = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
        cost = tf.reduce_mean(cost)

        # Regularized cost, L2 and dropout regularize the network
        if l2:
            regularizers = tf.nn.l2_loss(weights['out'])
            cost = cost + tf.reduce_mean( regularizers*beta )

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    with tf.name_scope('Accuracy'):
        # Accuracy
        acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        acc_op = tf.reduce_mean(tf.cast(acc, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()


    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        # Training cycle
        for epoch in range(training_epochs):

            _, loss = sess.run([optimizer, cost], feed_dict={x: X,
                                                              y: labels,keep_prob:dropout})
            # Display some epochs losses
            if (epoch+1)%display_step== 0 and 1==0:

                loss, acc= sess.run([cost , acc_op],feed_dict={x: X,y: labels, keep_prob:1.})


                print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                      "{:.4f}".format(loss),' Training accuracy ',"{:.4f}".format(acc))


            if epoch > (training_epochs*0.5):
                lr *= decay_factor

        y_pred = tf.argmax(pred, 1)
        y_true = tf.argmax(y, 1)

        y_p, y_t = sess.run([y_pred, y_true], {x: X_test, y: y_test, keep_prob: 1.})

        target_names = ['No', 'Yes']

        if verbose == 1:
            print('Multilayer Perceptron Report:\n',classification_report(y_t, y_p, target_names=target_names))
            print('Number of test positives is: ', sum(y_t))
            print('Number of pred positives is: ', sum(y_p))

        recall = recall_score(y_t, y_p)
        precision = precision_score(y_t, y_p)
        accuracy = accuracy_score(y_t, y_p)
        score = [accuracy, recall, precision ,sum(y_p)]

        return np.reshape(np.array(score),[1,-1])

