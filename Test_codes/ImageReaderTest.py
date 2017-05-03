from Image_operation import *
from HyperParam import *

reader = _ImageBatchManager(MNIST_TRAIN_IMG_DIR, '1')
while 1:
    batch = reader.get_batch()
print()