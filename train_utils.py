import json
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

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


def tsne_visualization(x, labels=None, perplexity=30, iterations=4):
    # in case data has a very high dimensionality, let's reduce it
    # using PCA (denoising effect) as suggested by Hinton..
    # https://lvdmaaten.github.io/publications/misc/Supplement_JMLR_2008.pdf
    if x.shape[1] > 50:
        # truncated SVD for sparse data
        print("Applying PCA for denoising before t-sne")
        x = PCA(n_components=30).fit_transform(x)
        print("New data shape", x.shape)

    best_x = TSNE(n_components=2, perplexity=perplexity).fit(x)
    for i in range(5):
        z = TSNE(n_components=2, perplexity=perplexity).fit(x)
        if z.kl_divergence_ < best_x.kl_divergence_:
            best_x = z
    plt.scatter(best_x.embedding_[:, 0], best_x.embedding_[:, 1])
    if labels is not None:
        i = 0
        for x, y in zip(best_x.embedding_[:, 0], best_x.embedding_[:, 1]):
            plt.text(x, y, str(labels[i]), color="red", fontsize=12)
            i += 1

    plt.show()