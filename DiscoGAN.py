import os

from Image_operation import ImageProvider
from Model_2 import *
from OptimizerGenerator import *
import tensorflow as tf
from Model_2.Construct import *
from placeholder_to_tensor import *


# 关闭tensorboard, 清空Event log
os.popen('killall tensorboard')
file_list = os.listdir(EVENT_DIR)
for file in file_list:
    os.remove(os.path.join(EVENT_DIR, file))

# ImageProvider
mnistProvider = ImageProvider(MNIST_TRAIN_IMG_DIR, ['0'], BATCH_SIZE, TOTAL_MNIST_BUFF_SIZE_GB)
cervixProvider = ImageProvider(CERVIX_TRAIN_IMG_DIR, ['Type_1'], BATCH_SIZE,
                               TOTAL_CERVIX_BUFF_SIZE_GB)

# 读入cervix图片, 从0-255转化为0-1
cer_real_ph = tf.placeholder(tf.float32, shape=[BATCH_SIZE, CERVIX_IMG_SIZE, CERVIX_IMG_SIZE, 3], name='cer_real_ph')
cer_real_tensor = get_cervix_tensor(cer_real_ph)

# 读入mnist, 从0-255转化为0-1
mnist_real_ph = tf.placeholder(tf.float32, shape=[BATCH_SIZE, MNIST_IMG_SIZE, MNIST_IMG_SIZE], name='mnist_real_ph')
mnist_real_tensor = get_mnist_tensor(mnist_real_ph)

# 建图
G_AB_A = Generator(cer_real_tensor,construct_AB, 'G_AB', 'A')
G_BA_A = Generator(G_AB_A.generated_img,construct_BA, 'G_BA', 'A')
G_BA_B = Generator(mnist_real_tensor, construct_BA, 'G_BA', 'B', reuse=True)
G_AB_B = Generator(G_BA_B.generated_img,construct_AB, 'G_AB', 'B', reuse=True)

D_B_real = Discriminator(mnist_real_tensor,construct_D_B, 'D_B','real')
D_B_fake = Discriminator(G_AB_A.generated_img, construct_D_B, 'D_B','fake', reuse=True)
D_A_real = Discriminator(cer_real_tensor, construct_D_A, 'D_B','real')
D_A_fake = Discriminator(G_BA_B.generated_img, construct_D_A, 'D_B','fake', reuse=True)

# Optimizer
opt_G_AB = GeneratorOptimizer(G_AB_A, D_B_fake, cer_real_tensor, G_BA_A.generated_img, LEARNIGN_RATE, 'opt_G_AB')
opt_G_BA = GeneratorOptimizer(G_BA_A, D_A_fake, mnist_real_tensor, G_AB_B.generated_img, LEARNIGN_RATE, 'opt_G_BA')
opt_D_B = DiscriminatorOpimizer(D_B_real,D_B_fake,LEARNIGN_RATE,'opt_D_B')
opt_D_A = DiscriminatorOpimizer(D_A_real,D_A_fake,LEARNIGN_RATE,'opt_D_A')

# Merge
merge_G_AB = tf.summary.merge(opt_G_AB.summaries + G_AB_A.summaries)
merge_G_BA = tf.summary.merge(opt_G_BA.summaries + G_BA_B.summaries)
merge_D_A = tf.summary.merge(opt_D_A.summaries + D_A_real.summaries + D_A_fake.summaries)
merge_D_B = tf.summary.merge(opt_D_B.summaries + D_B_real.summaries + D_B_fake.summaries)

# -------------------
saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
global_step = 1
# config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    file_writer = tf.summary.FileWriter(EVENT_DIR, graph=sess.graph)

    while 1:
        if global_step <= 100:
            n_disc = N_DISC_1
            write_summary_step = WRITE_SUMMARY_STEP_1
            save_step = SAVE_STEP_1
        else:
            n_disc = N_DISC_2
            write_summary_step = WRITE_SUMMARY_STEP_2
            save_step = SAVE_STEP_2

        cervix_real = cervixProvider.get_image_batch(type=0)
        mnist_real = mnistProvider.get_image_batch(type=0)  # mnist_0

        feed_dict = {cer_real_ph: cervix_real,
                     mnist_real_ph: mnist_real}

        # write_summary_g_flag
        write_summary_g_flag = True if (global_step % write_summary_step == 0) else False

        for iter_disc in range(n_disc):
            write_summary_d_flag = True if (write_summary_g_flag and (iter_disc + 1) % n_disc == 0) else False

            fetch = [opt_D_A.optimizer,opt_D_B.optimizer]

            if write_summary_g_flag and (iter_disc + 1) % n_disc == 0:
                fetch.append(merge_D_A)
                fetch.append(merge_D_B)

            result_d = sess.run(fetch, feed_dict=feed_dict)

            if write_summary_d_flag is True:
                file_writer.add_summary(result_d[2], global_step=global_step)
                file_writer.add_summary(result_d[3], global_step=global_step)
                file_writer.flush()

            if (iter_disc + 1) % 10 == 0:
                print('discriminator:\titer:' + str(iter_disc + 1))

        # g
        result_g = sess.run([opt_G_AB.optimizer, opt_G_BA.optimizer, merge_G_AB, merge_G_BA] if write_summary_g_flag is True else [opt_G_AB.optimizer, opt_G_BA.optimizer],
                            feed_dict=feed_dict)
        if write_summary_g_flag is True:
            file_writer.add_summary(result_g[2], global_step=global_step)
            file_writer.add_summary(result_g[3], global_step=global_step)
            file_writer.flush()
        print('generator: finished\n----------------------------------------')

        if global_step % DISPLAY_STEP == 0:
            print("Step:" + str(global_step) + '\tfinished!\n========================================')
        if global_step % save_step == 0:
            saver.save(sess, os.path.join(SAVE_PATH, 'DiscoGAN.save'), global_step=global_step)
            print('-------\n|saved|\n-------')
        global_step += 1
