'''
By adidinchuk park. adidinchuk@gmail.com.
https://github.com/adidinchuk/tf-support-vector-machines
'''

import tensorflow as tf
import utils as utl
import numpy as np
from tensorflow.python.framework import ops


class Network:

    def __init__(self, features, classes, gamma=-50.0):

        ops.reset_default_graph()
        # set params
        self.features = features
        self.classes = classes
        self.batch_size = None
        self.gamma = gamma

        # placeholders
        self.inputs = tf.placeholder(shape=[None, self.features], dtype=tf.float32)
        self.target = tf.placeholder(shape=[self.classes, None], dtype=tf.float32)
        self.grid = tf.placeholder(shape=[None, self.features], dtype=tf.float32)

        # model output, loss and b (alpha)
        self.model_output, self.loss, self.alpha = None, None, None

        # performance, tracking & init
        self.prediction, self.accuracy, self.optimization, self.training_step, self.init\
            = None, None, None, None, None

        # init session
        self.session = tf.Session()

        # support vector data for generating predictions post training
        self.support_vector_data = None

    def train(self, inputs, targets, lr=1, batch_size=30, epochs=100, plot=False, kernel='linear'):

        self.batch_size = batch_size
        # init the kernel
        self.set_kernel(kernel)

        # set optimization method (Gradient Descent)
        self.optimization = tf.train.GradientDescentOptimizer(lr)
        self.training_step = self.optimization.minimize(self.loss)
        self.init = tf.global_variables_initializer()
        self.session.run(self.init)

        # set training data
        train_inputs, train_target = inputs, targets

        # performance tracking
        train_loss_result, train_accuracy_result = [], []

        # for each epoch
        for i in range(epochs):

            # generate random indexes for each batch
            batch_index = np.random.choice(len(train_inputs), size=batch_size)
            self.session.run(self.training_step, feed_dict={self.inputs: train_inputs[batch_index],
                                                            self.target: train_target[:, batch_index]})
            # if plotting, record every epoch
            if plot:
                # record accuracy
                train_accuracy, train_loss = self.generate_step_tracking_data(
                    train_inputs[batch_index], train_target[:, batch_index])
                train_accuracy_result.append(train_accuracy)
                train_loss_result.append(train_loss)

            if (i+1) % (epochs / 5) == 0:
                # if not plotting, get intermittent accuracy and loss
                if not plot:
                    # record accuracy
                    train_accuracy, train_loss = self.generate_step_tracking_data(
                        train_inputs[batch_index], train_target[:, batch_index])
                utl.print_progress(i, epochs, train_loss, train_accuracy)

        # plot results
        if plot:
            if not self.features == 2:
                print('Plotting only supported for 2 feature data sets... skipping output')
            else:
                utl.plot_loss(train_loss_result)
                utl.plot_accuracy(train_accuracy_result)
                grid = utl.generate_grid(train_inputs)
                grid_predictions = self.session.run(self.prediction, feed_dict={self.inputs: train_inputs[batch_index],
                                                                                self.target: train_target[:, batch_index],
                                                                                self.grid: grid})
                # plot the result grid
                utl.plot_result(grid_predictions, inputs, targets)

        # commit data points for the last support vectors used
        self.support_vector_data = [train_inputs[batch_index], train_target[:, batch_index]]

    # linear kernel (X * U)
    def linear_kernel(self):
        # variables
        kernel = tf.matmul(self.inputs, tf.transpose(self.inputs))
        prediction_kernel = tf.matmul(self.inputs, tf.transpose(self.grid))
        self.complete_graph(kernel, prediction_kernel)

    # gaussian kernel (e ^ [ | X * U | ^ 2 ] )
    def gaussian_kernel(self):
        distribution = tf.reshape(tf.reduce_sum(tf.square(self.inputs), 1), [-1, 1])
        square_distributions = tf.add(tf.subtract(distribution, tf.multiply(2., tf.matmul(
            self.inputs, tf.transpose(self.inputs)))), tf.transpose(distribution))
        kernel = tf.exp(tf.multiply(tf.constant(self.gamma), tf.abs(square_distributions)))

        # square X and U vectors
        Xsqrt = tf.reshape(tf.reduce_sum(tf.square(self.inputs), 1), [-1, 1])
        Usqrt = tf.reshape(tf.reduce_sum(tf.square(self.grid), 1), [-1, 1])

        # [ | X * U | ^ 2 ]
        prediction_square_distribution = tf.add(tf.subtract(
            Xsqrt, tf.multiply(2., tf.matmul(self.inputs, tf.transpose(self.grid)))), tf.transpose(Usqrt))

        # (e ^ [ | X * U | ^ 2 ] )
        prediction_kernel = tf.exp(tf.multiply(tf.constant(self.gamma), tf.abs(prediction_square_distribution)))
        self.complete_graph(kernel, prediction_kernel)

    # finalize remainder fo the graph
    def complete_graph(self, kernel, prediction_kernel):
        self.alpha = tf.Variable(tf.random_normal(shape=[self.classes, self.batch_size]))
        alpha_vector_cross = tf.matmul(tf.transpose(self.alpha), self.alpha)
        target_vector_cross = self.reshape_matmul(self.target)
        self.loss = tf.reduce_sum(tf.negative(tf.subtract(tf.reduce_sum(self.alpha), tf.reduce_sum(
            tf.multiply(kernel, tf.multiply(alpha_vector_cross, target_vector_cross)), [1, 2]))))
        self.loss = tf.reduce_sum(tf.negative(tf.subtract(tf.reduce_sum(self.alpha), tf.reduce_sum(
            tf.multiply(kernel, tf.multiply(alpha_vector_cross, target_vector_cross)), [1, 2]))))
        prediction_output = tf.matmul(tf.multiply(self.target, self.alpha), prediction_kernel)
        self.prediction = tf.arg_max(prediction_output - tf.expand_dims(tf.reduce_mean(prediction_output, 1), 1), 0)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(self.prediction, tf.arg_max(self.target, 0)), tf.float32))

    # set the required kernel
    def set_kernel(self, kernel):
        if kernel == 'linear':
            self.linear_kernel()
        if kernel == 'gaussian':
            self.gaussian_kernel()

        # HELPERS #
    def set_params(self, gamma=None, features=None, refresh_graph_var=False):
        if gamma is not None:
            self.gamma = gamma
        if features is not None:
            self.features = features
        if refresh_graph_var:
            self.refresh_vars()

    # helped function to reshape matrix and multiply
    def reshape_matmul(self, matrix):
        first_vector = tf.expand_dims(matrix, 1)
        second_vector = tf.reshape(first_vector, [self.classes, self.batch_size, 1])
        return tf.matmul(second_vector, first_vector)

    def refresh_vars(self):
        # init variables
        self.init = tf.global_variables_initializer()
        self.session.run(self.init)

    def generate_step_tracking_data(self, inputs, targets):
        accuracy = self.session.run(self.accuracy, feed_dict={self.inputs: inputs, self.target: targets, self.grid: inputs})
        loss = self.session.run(self.loss, feed_dict={self.inputs: inputs, self.target: targets})
        return accuracy, loss

    def predict(self, features):
        return self.session.run(self.prediction, feed_dict={self.grid: np.array(features),
                                                            self.target: self.support_vector_data[1],
                                                            self.inputs: self.support_vector_data[0]})
