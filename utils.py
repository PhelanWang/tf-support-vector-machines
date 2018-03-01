'''
By adidinchuk park. adidinchuk@gmail.com.
https://github.com/adidinchuk/tf-support-vector-machines
'''

import numpy as np
import matplotlib.pyplot as plt


def test_train_split(inputs, target, test_size=0.5):
    train_size = int(len(inputs) - (len(inputs) * test_size))
    train_index = np.random.choice(len(inputs), size=train_size)
    test_index = np.array(list(set(range(len(inputs))) - set(train_index)))
    train_inputs = inputs[train_index]
    train_target = target[train_index]
    test_inputs = inputs[test_index]
    test_target = target[test_index]
    return train_inputs, train_target, test_inputs, test_target


def plot_loss(loss):
    fig, ax = plt.subplots()
    plt.title('Training Loss')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss (Cost Function)')
    plt.plot(loss, 'b-', label='Loss per generation')
    ax.legend(loc='upper right', shadow=True)
    plt.grid()
    plt.show()


def plot_accuracy(accuracy):
    fig, ax = plt.subplots()
    plt.title('Training Accuracy')
    plt.xlabel('Epoch #')
    plt.ylabel('Accuracy (%)')
    plt.plot(accuracy, 'b-', label='Generation accuracy')
    ax.legend(loc='lower right', shadow=True)
    plt.grid()
    plt.show()


def print_progress(current_epoch, epochs, train_loss, train_accuracy):
    print('Epoch #' + str(current_epoch + 1) + ' of ' + str(epochs))
    print('Training data loss: ', train_loss)
    print('Training data accuracy: ', train_accuracy)


def plot_result(grid_predictions, inputs, targets):
    x_min, x_max = inputs[:, 0].min() - 1, inputs[:, 0].max() + 1
    y_min, y_max = inputs[:, 1].min() - 1, inputs[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    grid_predictions = grid_predictions.reshape(xx.shape)
    tmp = np.transpose(targets)
    class1_x = [x[0] for i, x in enumerate(inputs) if tmp[i][0] == 1]
    class1_y = [x[1] for i, x in enumerate(inputs) if tmp[i][0] == 1]
    class2_x = [x[0] for i, x in enumerate(inputs) if tmp[i][1] == 1]
    class2_y = [x[1] for i, x in enumerate(inputs) if tmp[i][1] == 1]
    # get third data cluster if relevant
    if len(targets) == 3:
        class3_x = [x[0] for i, x in enumerate(inputs) if tmp[i][2] == 1]
        class3_y = [x[1] for i, x in enumerate(inputs) if tmp[i][2] == 1]

    # Plot points and grid
    plt.contourf(xx, yy, grid_predictions, cmap=plt.cm.Paired, alpha=0.8)
    plt.plot(class1_x, class1_y, 'ro', label='Class 1')
    plt.plot(class2_x, class2_y, 'kx', label='Class 2')
    # plot third data cluster if relevant
    if len(targets) == 3:
        plt.plot(class3_x, class3_y, 'gv', label='Class 3')
    plt.title('Gaussian SVM Results')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='lower right')
    plt.show()


def generate_grid(inputs):
    x_min, x_max = inputs[:, 0].min() - 1, inputs[:, 0].max() + 1
    y_min, y_max = inputs[:, 1].min() - 1, inputs[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    return grid_points
