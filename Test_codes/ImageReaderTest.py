from Image_Operate import *
from HyperParam import *

reader = ImageBatchManager(MNIST_TRAIN_IMG_DIR,'1')
while 1:
    batch = reader.get_batch()
print()