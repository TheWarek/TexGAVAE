from _tracemalloc import start

import numpy as np
import os
import random
import matplotlib.pyplot as plt

from array import array

from skimage import transform
import sklearn.preprocessing as prep


def resize_batch_images(batch: list, image_size: tuple) -> np.ndarray:
    return np.asarray([transform.resize(image, image_size, mode="reflect") for image in batch])


class ImageObjects:
    def __init__(self):
        self.objects = []
        self.name = ''

class ObjectSuperpixels:
    def __init__(self):
        self.superpixels = []
        self.labels = []

class TexDAT:

    class ObjectDataContainer:
        def __init__(self):
            self.name = ""
            self.paths = []

    def __init__(self, path, batch_size=128, image_size=None, use_grayscale=False):
        self.batch_size = batch_size
        self.train = TexDAT.train(path, batch_size, image_size, use_grayscale)
        self.test = TexDAT.test(path, batch_size, image_size, use_grayscale)
        self.only_paths = False

    def set_path(self, path):
        if os.path.exists(os.path.abspath(path)):
            self.train.abs_path = os.path.abspath(path) + "\\train"
            self.test.abs_path = os.path.abspath(path) + "\\test"

    def load_data(self, only_paths: bool = False):
        self.read_data_to_array(self.train.abs_path, train=True, only_paths=only_paths)
        self.read_data_to_array(self.test.abs_path, test=True, only_paths=only_paths)
        self.only_paths = only_paths
        return True

    @staticmethod
    def read_segment_file(abspath: str, texid: int = None):
        if not os.path.exists(abspath):
            return
        try:
            with open(abspath, "rb") as bf:
                channels = int.from_bytes(bf.read(4), byteorder="little", signed=False)
                objidx = int.from_bytes(bf.read(4), byteorder="little", signed=False)
                texs = int.from_bytes(bf.read(4), byteorder="little", signed=False)
                if texid is None:
                    texid = random.randint(0,texs)
                for i in range(texs):
                    rows = int.from_bytes(bf.read(4), byteorder="little", signed=False)
                    cols = int.from_bytes(bf.read(4), byteorder="little", signed=False)

                    data = np.empty(shape=[rows, cols, channels], dtype=np.ubyte)

                    bf.readinto(data.data)
                    data = (data/256).astype(np.float32)
                    if i == texid:
                        bf.close()
                        return data

        except IOError:
            print("File {:s} does not exist".format(abspath))

    def read_data_to_array(self, abspath=None, only_paths: bool = False, train=False, test=False):
        if not os.path.exists(abspath):
            return
        if not (train or test):
            return
        print("Loading from \"" + abspath + "\"")
        files = os.listdir(abspath)
        files_as_objects = {}
        for file in files:
            if only_paths: # reads only paths - runtime loading files
                name = file[file.rfind("/") + 1:file[0:file.rfind("_")].rfind("_")]
                if files_as_objects.get(name):
                    files_as_objects[name].paths.append(os.path.join(abspath, file))
                else:
                    files_as_objects[name] = self.ObjectDataContainer()
                    files_as_objects[name].name = name
                    files_as_objects[name].paths.append(os.path.join(abspath, file))
                img_objs = ImageObjects()
                img_objs.name = file
            else: # preloads all images to RAM
                try:
                    with open(os.path.join(abspath, file), "rb") as bf:
                        channels = int.from_bytes(bf.read(4), byteorder="little", signed=False)
                        objs = int.from_bytes(bf.read(4), byteorder="little", signed=False)
                        for o in range(objs):
                            obj_sups = ObjectSuperpixels()
                            sups = int.from_bytes(bf.read(4), byteorder="little", signed=False)
                            for s in range(sups):
                                if file.endswith('.supl'):
                                    label = int.from_bytes(bf.read(4), byteorder="little", signed=False)
                                    obj_sups.labels.append(label)
                                rows = int.from_bytes(bf.read(4), byteorder="little", signed=False)
                                cols = int.from_bytes(bf.read(4), byteorder="little", signed=False)
                                data = np.empty(shape=[rows, cols, channels], dtype=np.ubyte)
                                bf.readinto(data.data)
                                # data = (data / 255).astype(np.float32)
                                data = prep.scale(data.reshape((rows * cols * channels))).reshape(
                                    (rows, cols, channels))
                                # plt.imshow(data)
                                obj_sups.superpixels.append(data)
                            if train:
                                self.train.append(obj_sups)
                            elif test:
                                self.test.append(obj_sups)
                            img_objs.objects.append(obj_sups)
                        bf.close()
                    if train:
                        self.train.add_object(img_objs)
                    elif test:
                        self.test.add_object(img_objs)
                except IOError:
                    print("File {:s} does not exist".format(os.path.join(abspath, file)))

        if train:
            self.train.data = np.array(self.train.data)
            self.train.objectsPaths = files_as_objects
            if only_paths:
                print(str(len(self.train.objectsPaths)) + " train objects loaded")
            else:
                print(str(len(self.train.data)) + " train objects loaded")
        if test:
            self.test.objectsPaths = files_as_objects
            self.test.data = np.array(self.test.data)
            if only_paths:
                print(str(len(self.test.objectsPaths)) + " test objects loaded")
            else:
                print(str(len(self.test.data)) + " test objects loaded")

    @staticmethod
    def next_classic_batch_from_paths(paths: dict, batch_size: int = None, image_size = None):
        batch = batch_size
        if batch_size == None:
            return []
        items = np.array(list(paths.values()))
        classes = np.random.choice(items, batch, False)
        selected_paths = [random.choice(c.paths) for c in classes]

        selected = [TexDAT.read_segment_file(p) for p in selected_paths]

        if image_size:
            selected = resize_batch_images(selected, image_size)

        selected = [prep.scale(s.reshape((s.shape[0] * s.shape[1] * s.shape[2]))).reshape((s.shape[0], s.shape[1], s.shape[2])) for s in selected.data]

        return selected

    @staticmethod
    def next_similarity_batch_from_paths(paths: dict, batch_size: int = None, image_size=None, return_paths: bool = False):
        batch = batch_size
        if batch_size == None:
            return []
        if batch > len(paths):
            batch = (len(paths) >> 1) << 1
        neg_size = (int(batch) >> 1) << 1
        pos_size = int(batch) >> 1

        items = np.array(list(paths.values()))
        neg_classes = np.random.choice(items, neg_size, False)
        pos_classes = np.random.choice(items, pos_size, False)
        neg_s = [random.choice(c.paths) for c in neg_classes]
        neg_1 = neg_s[0:(neg_size >> 1)]
        neg_2 = neg_s[(neg_size >> 1):neg_size]
        pos_s = np.array([random.sample(c.paths, 2) for c in pos_classes])

        batch_1_paths = np.concatenate((neg_1,pos_s[:,0]))
        batch_2_paths = np.concatenate((neg_2,pos_s[:,1]))

        labels = np.zeros(batch, dtype=np.float32)
        labels[(batch>>1):batch] = 1

        batch_1 = [TexDAT.read_segment_file(p) for p in batch_1_paths]
        batch_2 = [TexDAT.read_segment_file(p) for p in batch_2_paths]

        if image_size:
            batch_1 = resize_batch_images(batch_1, image_size)
            batch_2 = resize_batch_images(batch_2, image_size)

        if return_paths:
            return batch_1, batch_2, labels, batch_1_paths, batch_2_paths
        else:
            return batch_1, batch_2, labels


    class train:

        def __init__(self, path, batch_size=128, start_step=0, max_steps=5000, size=None, grayscale=False):
            self.abs_path = os.path.abspath(path) + "\\train"
            self.batch_size = batch_size
            self.start_step = start_step
            self.max_steps = max_steps
            self.images = []
            self.data = []
            self.image_size = size
            self.use_grayscale = grayscale
            self.objectsPaths = {}

        def append(self, o):
            self.data.append(o)

        def add_object(self, o):
            self.images.append(o)

    class test:
        def __init__(self, path, batch_size=128, size=None, grayscale=False):
            self.abs_path = os.path.abspath(path) + "\\test"
            self.batch_size = batch_size
            self.data = []
            self.images = []
            self.image_size = size
            self.use_grayscale = grayscale
            self.objectsPaths = []

        def append(self, o):
            self.data.append(o)

        def add_object(self, o):
            self.images.append(o)


