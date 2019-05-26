import json
import matplotlib.pyplot as plt
import os


def train_info_to_json(hist:dict, filename):
    dir = "json_results"
    with open(dir + os.sep + filename, 'w') as f:
        json.dump(hist, f)
    print("{} correctly dumped to disk".format(filename))


def select_best_models(dir):
    # filename template is {firstlayer_length}-{n_layers}-layers_dae-layer_{}.json
    # filename template is {encoding_dim}_dim-shallow_ae.json
    min = 10000.0
    for filename in os.listdir(dir):
        if 'shallow_ae' in filename:
            # load shallow ae model results
            pass
        elif 'dae-layer' in filename:
            # load deep model results
            pass

def plot(y, x=None, labels=None, title=None, style=None):
    if x:
        plt.plot(x, y)
    else:
        plt.plot(y)

    if labels:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
    plt.title(title)

    plt.show()


def show_imgs(test_images, reconstruction, n):
    plt.figure(num=None, figsize=(20, 4), dpi=80)
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(test_images[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstruction[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()