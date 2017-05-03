EVENT_DIR = '/home/elijha/PycharmProjects/Cervix_Classification_GAN/Events/4_24'
SAVE_PATH = '/home/elijha/PycharmProjects/Cervix_Classification_GAN/Saves/4_24'

CERVIX_TRAIN_IMG_DIR = '/home/elijha/Documents/Data/Cervix/train_converted'
MNIST_TRAIN_IMG_DIR = '/home/elijha/Documents/Data/MNIST'

CERVIX_IMG_SIZE = 224
MNIST_IMG_SIZE = 28

MAX_PIXEL_VALUE = 255.0
BATCH_SIZE = 64
DISPLAY_STEP = 1
VALIDATE_STEP = 1

IS_WGAN = True

TOTAL_CERVIX_BUFF_SIZE_GB = 4
TOTAL_MNIST_BUFF_SIZE_GB = 2

LEARNIGN_RATE = 0.01

if IS_WGAN:
    CLIP = 0.02
    N_DISC_1 = 100
    N_DISC_2 = 10
    WRITE_SUMMARY_STEP_1 = 1
    WRITE_SUMMARY_STEP_2 = 20
    SAVE_STEP_1 = 50
    SAVE_STEP_2 = 200
else:
    N_DISC_1 = 5
    N_DISC_2 = 1
    WRITE_SUMMARY_STEP_1 = 10
    WRITE_SUMMARY_STEP_2 = 50
    SAVE_STEP_1 = 200
    SAVE_STEP_2 = 500

OPTIMIZER_SUMMARIES = [
    "learning_rate",
    "loss",
    # "gradients",
    "gradient_norm",
]
