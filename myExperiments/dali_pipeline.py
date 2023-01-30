from nvidia.dali import pipeline_def, fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import time
import os
import numpy as np

def gen_cifar10_npy():
    # the cifar10 dataset is already saved in numpy array, not a image format
    train_data = datasets.CIFAR10('./data', train=True, download=True)
    validation_data = datasets.CIFAR10('./data', train=False, download=True)

    cifar10_x_train = train_data.data
    cifar10_y_train = train_data.targets
    cifar10_x_test = validation_data.data
    cifar10_y_test = validation_data.targets

    np.save("./data/cifar10_x_train", cifar10_x_train)
    np.save("./data/cifar10_y_train", cifar10_y_train)
    np.save("./data/cifar10_x_test", cifar10_x_test)
    np.save("./data/cifar10_y_test", cifar10_y_test)

# one time function
gen_cifar10_npy()

data_dir = "./data"

data_path = os.path.join(data_dir, 'cifar10_x_train.npy')
label_path = os.path.join(data_dir, 'cifar10_y_train.npy')
data_npy = np.load(data_path)
label_npy = np.load(label_path)

class ExternalInputIterator(object):
    def __init__(self, data_npy, label_npy, batch_size):
        self.data = data_npy
        self.label = label_npy
        self.batch_size = batch_size

    def __iter__(self):
        self.i = 0
        self.n = len(self.label)
        return self

    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            img, label = self.data[self.i], self.label[self.i]
            batch.append(img)
            labels.append(label)
            self.i = (self.i + 1) % self.n   # this create an infinity loop, the ends is set in DALIGenericIterator size
        return (np.array(batch), np.array(labels))

batch_size = 32
eii = ExternalInputIterator(data_npy, label_npy, batch_size)

@pipeline_def(batch_size=batch_size, num_threads=4, device_id=0)
def pipe1():
    # step 1) read data
    # device='gpu' is not supported for this fn, gpu direct read requires gds, not enabled in the container
    data, label = fn.external_source(device='cpu', source=eii, num_outputs=2) # must define num_outputs=2, otherwise mix data and label tensor, and complain tensor dim mismatch in a batch
    # step 2) crop, mirror, normalization
    data = fn.crop_mirror_normalize(
        data.gpu(),  #send data to gpu, if dont add .gpu(), and remove device='gpu', execute on cpu, and current setting cpu is faster
        crop_h=32,
        crop_w=32,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        mirror=fn.random.coin_flip(),
        device='gpu'
        )
    # add more steps/fns if you like
    return data, label

train_data = DALIGenericIterator(
        [pipe1()],
        ['data', 'label'],
        size=len(label_npy)   # must set the size, otherwise become infinity
        )

start = time.time()
for i, sample in enumerate(train_data):
    x = sample[0]['data']
    y = sample[0]['label']
    #break
print("total time {}".format(time.time()-start)) # first time with pipeline build, time is long

