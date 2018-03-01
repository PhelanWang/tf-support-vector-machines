'''
By adidinchuk park. adidinchuk@gmail.com.
https://github.com/adidinchuk/tf-support-vector-machines
'''

import hyperparams as hp


# breast-cancer-wisconsin.data
def load_data(name):

    file_name = hp.data_dir + '\\' + name

    with open(file_name) as fp:
        lines = fp.read().split("\n")

    data = [line.split(',') for line in lines]
    return data


def abalone_gender_to_int(data):
    outputs = [0 if row[:1][0] == 'I' else 1 if row[:1][0] == 'M' else 2 for row in data]
    return outputs
