
from keras.models import load_model
import keras.backend as K
from data_loader import TexDAT
from data_loader import resize_batch_images, normalize_batch_images
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as pyplot

DATA_PATH = 'E:/Datasets/texdat'
batch_size = 128
patch_size = 160

m = batch_size
n_z = 128


def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(m, n_z), mean=0., stddev=1.)
    # sampling from Z~N(μ, σ^2) is the same as sampling from μ + σX, X~N(0,1)
    return mu + K.exp(log_sigma / 2) * eps


model = load_model('../enc.h5')
sess = K.get_session()


texdat = TexDAT(DATA_PATH, batch_size)
texdat.load_data(only_paths=True)


sorted_list = list(texdat.train.objectsPaths.items())
sorted_list.sort()

batch_test = []
# indices = [31, 103, 137, 194]
indices = [46, 46, 137, 194]
for ind in indices:
    for i in range(int(batch_size / 4)):
        #batch_test.append(self.texdat.read_image_patch(sorted_list[ind][1].paths[0], patch_size=self.patch_size))
        batch_test.append(texdat.read_segment(sorted_list[ind][1].paths[0]))
batch_test = resize_batch_images(batch_test, (patch_size, patch_size))
batch_test = normalize_batch_images(batch_test, 'zeromean')


mu, log_sigma = model.predict(batch_test)

# create samples to plot
samples = sample_z([mu, log_sigma])

samples = samples.eval(session=sess)
#time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
tsne_results = tsne.fit_transform(samples)

#print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

#pyplot.scatter(tsne_results[:, 0], tsne_results[:, 1], 20, labels)

colors = pyplot.cm.rainbow(np.linspace(0, 1, 10))
zero = pyplot.scatter(tsne_results[0:int(batch_size / 4), 0],
                    tsne_results[0:int(batch_size / 4), 1],
                    18, colors[0])
one = pyplot.scatter(tsne_results[int(batch_size / 4): int(2 * batch_size / 4), 0],
                    tsne_results[int(batch_size / 4): int(2* batch_size / 4), 1],
                    18, colors[1])
two = pyplot.scatter(tsne_results[int(2 * batch_size / 4): int(3 * batch_size / 4), 0],
                    tsne_results[int(2 * batch_size / 4): int(3 * batch_size / 4), 1],
                    18, colors[2])
three = pyplot.scatter(tsne_results[int(3 * batch_size / 4): int(4 * batch_size / 4), 0],
                    tsne_results[int(3 * batch_size / 4): int(4 * batch_size / 4), 1],
                    18, colors[3])

bottom, top = pyplot.ylim()  # return the current ylim
pyplot.ylim((bottom-15, top))  # set the ylim to bottom, top
pyplot.legend((zero, one, two, three),
           ('0', '1', '2', '3'),
           scatterpoints=1,
           loc='lower left',
           ncol=5)
pyplot.show()