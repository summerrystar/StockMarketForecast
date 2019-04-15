# The implementation of LSTM. Takes input of class model_data

import tensorflow as tf

# TODO: need to fix levels of the functions
# TODO: testing_lstm() is just copied

# constants
INPUT_SIZE = 1
TARGET_SIZE = 1
NUM_STEPS = 30
INIT_LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = 0.99
BATCH_SIZE = 64
KEEP_PROB = 0.8
LSTM_SIZE = 128
NUM_LAYERS = 1
INIT_EPOCH = 5
MAX_EPOCH = 100
VECTOR_SIZE = 7

def create_lstm_graph ():
    # Initialize graph
    tf.reset_default_graph()
    lstm_graph=tf.Graph()

    # Define the graph
    inputs  = tf.placeholder(tf.float32, [None, INPUT_SIZE, VECTOR_SIZE])
    targets = tf.placeholder(tf.float32, [None, TARGET_SIZE])
    learning_rate = tf.placeholder(tf.float32, None)

    lstm_cell = tf.contrib.rnn.LSTMcell(LSTM_SIZE, state_is_tuple=True, activation=tf.nn.tanh)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=KEEP_PROB, seed=42) # use seed to have the same dropouts each time
    lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * NUM_LAYERS, state_is_tuple=True)

    # ref: https://danijar.com/introduction-to-recurrent-networks-in-tensorflow/
    # search for keyword "gather"
    output, _ = tf.nn.dynamic_run(lstm_cell, inputs, dtype=tf.float32)
    output = tf.transpose(output, [1, 0, 2]) # optionally use time_major=True to avoid this transpose. But may need to change input shape
    last = tf.gather( output, int(output.get_shape()[0])-1 )

    weights = tf.Variable(tf.truncated_normal([LSTM_SIZE, INPUT_SIZE]))
    biases = tf.Variable(tf.constant(0.01, shape=[INPUT_SIZE]))
    prediction = tf.matmul(last, weights) + biases
    
    loss = tf.reduce_mean(tf.sqare(prediction - targets))
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    minimize = optimizer.minimize(loss)

    return(lstm_graph)

def training_lstm (lstm_graph, x_train, y_train):
    # Create a function called "chunks", l is a vector, n is batch size:
    # chop l into batches of size
    def chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i+n]
    with tf.Session(graph=lstm_graph) as sess:
        tf.global_variables_initializer().run()
        for epoch_step in range(MAX_EPOCH):
            learning_rate = INIT_LEARNING_RATE * (LEARNING_RATE_DECAY ** max(float(epoch_step+1-INIT_EPOCH, 0.0)))
            for batch_X, batch_Y in zip( list(chunks(x_train, BATCH_SIZE)), list(chunks(y_train, BATCH_SIZE)) ):
                train_data_feed = {
                    inputs: batch_X, 
                    targets: batch_Y, 
                    learning_rate: learning_rate
                }
                training_loss, _ = sess.run([loss, minimize], train_data_feed)
                saver = tf.train.Saver()
                saver.save(sess, "/tmp/model.ckpt")

def testing_lstm (lstm_grapin, x_test, y_test):
    with tf.Session(graph=lstm_graph) as sess:
        saver=tf.train.Saver()
        saver.restore(sess, "/tmp/model.ckpt")
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("~/Downloads/model_log", sess.graph)
        writer.add_graph(sess.graph) 
        i = 0
        for batch_X, batch_Y in zip(list(chunks(X_TEST, 1)), list(chunks(Y_TEST, 1))):
        test_data_feed = {inputs: batch_X, targets: batch_Y, learning_rate: current_lr}
        summary1, summary2, summary3 = sess.run([prediction, targets, pred], test_data_feed)
        i +=1
        data_frame[0].append(i)
        data_frame[1].append(np.ravel(summary3))
        data_frame[2].append(np.ravel(summary2))
        
    fig, ax = plt.subplots(figsize=(16,9))

    data_frame[2] = np.multiply(data_frame[2],1)
    data_frame[1] = np.multiply(data_frame[1],1)

    ax.plot(DATES_TEST, data_frame[2], label="target")
    ax.plot(DATES_TEST, data_frame[1], label="prediction")
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
