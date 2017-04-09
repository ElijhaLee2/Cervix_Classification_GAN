from GAN_models import *
from Image_Operate import *
from other_function import *
import os

os.popen('killall tensorboard')
file_list = os.listdir(EVENT_DIR)
for file in file_list:
    os.remove(os.path.join(EVENT_DIR, file))

# 将数据集读入内存
cer_batch_mngr = ImageBatchManager(CERVIX_TRAIN_IMG_DIR,'Type_1')
mnist_batch_mngr = ImageBatchManager(MNIST_TRAIN_IMG_DIR,'1')

# 把cervix转为mnist
cer_ph = tf.placeholder(tf.float32, shape=[BATCH_SIZE, CERVIX_IMG_SIZE, CERVIX_IMG_SIZE, 3], name='cer_ph')
cer_img_tensor = cer_ph - tf.reduce_mean(cer_ph, axis=[0,1,2])
with tf.variable_scope('generator'):
    g_mnist = generator(cer_img_tensor)

# 读入mnist
mnist_ph = tf.placeholder(tf.float32, shape=[BATCH_SIZE, MNIST_IMG_SIZE, MNIST_IMG_SIZE], name='mnist_ph')
mnist_img_tensor = tf.expand_dims(mnist_ph,axis=3)

# 得到fw的评分
with tf.variable_scope('critic'):
    fw_g_mnist = critic(g_mnist)
with tf.variable_scope('critic', reuse=True):
    fw_mnist = critic(mnist_img_tensor)

# 梯度下降
optm_c, optm_g = get_optimizers(fw_g_mnist, fw_mnist, 0.02)

# 可视化
sum_g_z = tf.summary.image('g_z', g_mnist, max_outputs=BATCH_SIZE)
sum_image_fake = tf.summary.histogram('image_fake', g_mnist)
sum_image_real = tf.summary.histogram('image_real', mnist_img_tensor)

sum_fw_g_z = tf.summary.scalar('fw_g_z', tf.reduce_mean(fw_g_mnist))
sum_fw_x = tf.summary.scalar('fw_x', tf.reduce_mean(fw_mnist))

# sum_loss_c = tf.summary.scalar('loss_c', loss_c)
# sum_loss_g = tf.summary.scalar('loss_g', loss_g)
#
# sum_grad_c = [tf.summary.histogram('grad_'+g[1].name, g[0]) for g in grad_c]
# sum_grad_g = [tf.summary.histogram('grad_'+g[1].name, g[0]) for g in grad_g]
#
# sum_vars_c = [tf.summary.histogram(v.name, v) for v in vars_c]
# sum_vars_g = [tf.summary.histogram(v.name, v) for v in vars_g]

merge_all = tf.summary.merge_all()

# saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
global_step = 1

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    file_writer = tf.summary.FileWriter(EVENT_DIR, graph=sess.graph)

    while 1:
        # n_critic = N_CRITIC_1 if global_step <= 100 else N_CRITIC_2
        n_critic = 1
        flag_write_summary = True if global_step % WRITE_SUMMARY_STEP == 0 else False
        for i in range(n_critic):
            print('critics: ' + str(i))
            cer_batch = cer_batch_mngr.get_batch()
            mnist_batch = mnist_batch_mngr.get_batch()
            if flag_write_summary and i == n_critic - 1:
                _, merge = sess.run([optm_c, merge_all], feed_dict={cer_ph: cer_batch, mnist_ph: mnist_batch})
                file_writer.add_summary(merge, global_step=global_step)
                file_writer.flush()
            else:
                sess.run(optm_c, feed_dict={cer_ph: cer_batch, mnist_ph: mnist_batch})

        print('generator: ' )
        cer_batch = cer_batch_mngr.get_batch()
        mnist_batch = mnist_batch_mngr.get_batch()
        if flag_write_summary:
            _, merge = sess.run([optm_g, merge_all], feed_dict={cer_ph: cer_batch, mnist_ph: mnist_batch})
            # _, merge = sess.run([fw_g_mnist, merge_all], feed_dict={cer_ph: cer_batch, mnist_ph: mnist_batch})
            file_writer.add_summary(merge, global_step=global_step)
            file_writer.flush()
        else:
            sess.run(optm_g, feed_dict={cer_ph: cer_batch, mnist_ph: mnist_batch})
            # sess.run(fw_g_mnist, feed_dict={cer_ph: cer_batch, mnist_ph: mnist_batch})

        if global_step % DISPLAY_STEP == 0:
            print("Step:" + str(global_step))

        # if global_step % SAVE_STEP == 0:
        #     saver.save(sess, SAVE_STEP, global_step=global_step)

        global_step += 1


# time.sleep(5)
# os.popen('tensorboard --logdir=/home/elijha/PycharmProjects/WGAN/Events/3_12')
