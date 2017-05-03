from Image_operation import *
import numpy as np

batch_size = 8
image_batch_manager = _ImageBatchManager(CERVIX_TRAIN_IMG_DIR, 'Type_1', batch_size)
batch = image_batch_manager.get_batch()
batch = np.reshape(batch,[batch_size,-1])
aaa = np.dot(np.transpose(batch),batch)

e_value, e_vec = np.linalg.eig(aaa)

print()