import os

from Image_operation import ImageProvider
from Model import *
from OptimizerGenerator import *

# 关闭tensorboard, 清空Event log
os.popen('killall tensorboard')
# file_list = os.listdir(EVENT_DIR)
# for file in file_list:
#     os.remove(os.path.join(EVENT_DIR, file))

# ImageProvider
mnistProvider = ImageProvider(MNIST_TRAIN_IMG_DIR, ['0', '1', '4'], BATCH_SIZE, TOTAL_MNIST_BUFF_SIZE_GB)
cervixProvider = ImageProvider(CERVIX_TRAIN_IMG_DIR, ['Type_1', 'Type_2', 'Type_3'], BATCH_SIZE,
                               TOTAL_CERVIX_BUFF_SIZE_GB)

# 读入cervix图片, 从0-255转化为0-1
cer_real_ph = tf.placeholder(tf.float32, shape=[BATCH_SIZE, CERVIX_IMG_SIZE, CERVIX_IMG_SIZE, 3], name='cer_real_ph')
cer_real_tensor = cer_real_ph / MAX_PIXEL_VALUE

# 对Cervix进行像素归一化
ema = tf.train.ExponentialMovingAverage(decay=0.9)
[ave_batch, var_batch] = tf.nn.moments(cer_real_tensor, axes=[0, 1, 2])
with tf.control_dependencies([ema.apply([ave_batch, var_batch])]):
    channel_ave = ema.average(ave_batch)
    channel_std = tf.sqrt(ema.average(var_batch))
cer_real_tensor = (cer_real_tensor - channel_ave) / channel_std

# 读入Cervix的label
cervix_label_ph = tf.placeholder(tf.float32, [1, 3])

# 读入mnist, 从0-255转化为0-1
mnist_real_0_ph = tf.placeholder(tf.float32, shape=[BATCH_SIZE, MNIST_IMG_SIZE, MNIST_IMG_SIZE], name='mnist_real_0_ph')
mnist_real_0_tensor = mnist_real_0_ph / MAX_PIXEL_VALUE
mnist_real_0_tensor = tf.expand_dims(mnist_real_0_tensor, axis=3)
# -----
mnist_real_1_ph = tf.placeholder(tf.float32, shape=[BATCH_SIZE, MNIST_IMG_SIZE, MNIST_IMG_SIZE], name='mnist_real_1_ph')
mnist_real_1_tensor = mnist_real_1_ph / MAX_PIXEL_VALUE
mnist_real_1_tensor = tf.expand_dims(mnist_real_1_tensor, axis=3)
# -----
mnist_real_2_ph = tf.placeholder(tf.float32, shape=[BATCH_SIZE, MNIST_IMG_SIZE, MNIST_IMG_SIZE], name='mnist_real_2_ph')
mnist_real_2_tensor = mnist_real_2_ph / MAX_PIXEL_VALUE
mnist_real_2_tensor = tf.expand_dims(mnist_real_2_tensor, axis=3)

# 建图
generator = Generator(cer_real_tensor, 'G')
discriminator_0 = Discriminator(mnist_real_0_tensor, generator.generated_img, 'D_0')
discriminator_1 = Discriminator(mnist_real_1_tensor, generator.generated_img, 'D_1')
discriminator_2 = Discriminator(mnist_real_2_tensor, generator.generated_img, 'D_2')

# Optimizer
generator_optimizer = GeneratorOptimizer(generator, [discriminator_0, discriminator_1, discriminator_2],
                                         cervix_label_ph, LEARNIGN_RATE, 'G_optmzr')
dis_0_optimizer = DiscriminatorOpimizer(discriminator_0, LEARNIGN_RATE, 'D_0_optmzr')
dis_1_optimizer = DiscriminatorOpimizer(discriminator_1, LEARNIGN_RATE, 'D_1_optmzr')
dis_2_optimizer = DiscriminatorOpimizer(discriminator_2, LEARNIGN_RATE, 'D_2_optmzr')

# Merge
merge_g = tf.summary.merge(generator.summaries + generator_optimizer.summaries)
merge_d_0 = tf.summary.merge(discriminator_0.summary+dis_0_optimizer.summaries)
merge_d_1 = tf.summary.merge(discriminator_1.summary+dis_1_optimizer.summaries)
merge_d_2 = tf.summary.merge(discriminator_2.summary+dis_2_optimizer.summaries)

# -------------------
saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
global_step = 1
# config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,'/home/elijha/PycharmProjects/Cervix_Classification_GAN/Saves/4_24/BigGun.save-400')
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
        for real_type in range(3):
            # label
            cervix_label = [[0., 0., 0.]]
            cervix_label[0][real_type] = 1.
            # feed_dict
            cervix_real = cervixProvider.get_image_batch(type=real_type)
            mnist_real_0 = mnistProvider.get_image_batch(type=0)  # mnist_0
            mnist_real_1 = mnistProvider.get_image_batch(type=1)  # mnist_1
            mnist_real_2 = mnistProvider.get_image_batch(type=2)  # mnist_0

            feed_dict = {cer_real_ph: cervix_real,
                         cervix_label_ph: cervix_label,
                         mnist_real_0_ph: mnist_real_0,
                         mnist_real_1_ph: mnist_real_1,
                         mnist_real_2_ph: mnist_real_2, }

            # write_summary_g_flag
            write_summary_g_flag = True if (global_step % write_summary_step == 0) else False

            for iter_disc in range(n_disc):
                write_summary_d_flag = True if (write_summary_g_flag and (iter_disc + 1) % n_disc == 0) else False

                fetch = [[dis_0_optimizer.optimizer], [dis_1_optimizer.optimizer], [dis_2_optimizer.optimizer]]

                if write_summary_g_flag and (iter_disc + 1) % n_disc == 0:
                    if real_type == 0:
                        fetch[real_type].append(merge_d_0)
                    elif real_type == 1:
                        fetch[real_type].append(merge_d_1)
                    elif real_type == 2:
                        fetch[real_type].append(merge_d_2)

                results = [None, None, None]
                for i in range(3):
                    results[i] = sess.run(fetch[i], feed_dict=feed_dict)

                if write_summary_d_flag is True:
                    file_writer.add_summary(results[real_type][1], global_step=global_step)
                    file_writer.flush()

                if (iter_disc + 1) % 10 == 0:
                    print('discriminator:\ttype:' + str(real_type) + ',\titer:' + str(iter_disc + 1))

            # g
            result_g = sess.run([generator_optimizer.optimizer, merge_g] if write_summary_g_flag is True else [
                generator_optimizer.optimizer],
                                feed_dict=feed_dict)
            if write_summary_g_flag is True:
                file_writer.add_summary(result_g[1], global_step=global_step)
                file_writer.flush()
            print('generator:\ttype:' + str(real_type) + '\n----------------------------------------')

        # if global_step % VALIDATE_STEP == 0:
        #     valid_type = int(np.random.uniform(0, 3))
        #     valid_imgs = cervixProvider.get_image_batch(valid_type)
        #     feed_dict[cer_real_ph]= valid_imgs
        #     w_distances=sess.run([dis_0_optimizer.w_distance,dis_0_optimizer.w_distance,dis_0_optimizer.w_distance],feed_dict=feed_dict) # (3,batch_size,1)
        #     w_distances = np.reshape(w_distances,(3,BATCH_SIZE)) # (3,batch_size)
        #     logits = np.divide(1,w_distances)
        #     multiclass_loss = 0
        #     for i in range(3):
        #         if i == real_type:
        #             continue
        #         multiclass_loss += np.maximum(0, np.add(logits[i]-logits[real_type], 1))
        #     classification = np.argmax(logits,axis=0) # (1,batch_size)
        #     labels = np.ones(shape=[1,50],dtype=int) * valid_type
        #     t_and_f= np.equal(logits,labels)
        #     print('Multiclass loss:\t'+str(multiclass_loss) +'\nClassification accuracy:\t' + str(t_and_f.tolist().count(True)/len(t_and_f))+'=====================================================')

        if global_step % DISPLAY_STEP == 0:
            print("Step:" + str(global_step) + '\tfinished!\n========================================')
        # if global_step % save_step == 0:
        #     saver.save(sess, os.path.join(SAVE_PATH, 'BigGun.save'), global_step=global_step)
        #     print('-------\n|saved|\n-------')
        global_step += 1
