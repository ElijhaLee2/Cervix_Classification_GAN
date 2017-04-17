from GAN_models import *
from Image_operation import *
from other_functions import *
import os

os.popen('killall tensorboard')
file_list = os.listdir(EVENT_DIR)
for file in file_list:
    os.remove(os.path.join(EVENT_DIR, file))

# 将数据集读入内存
cer_batch_mngr = ImageBatchManager(CERVIX_TRAIN_IMG_DIR,'Type_1', BATCH_SIZE)
mnist_batch_mngr = ImageBatchManager(MNIST_TRAIN_IMG_DIR,'1',BATCH_SIZE)

# 把cervix转为mnist
cer_ph = tf.placeholder(tf.float32, shape=[BATCH_SIZE, CERVIX_IMG_SIZE, CERVIX_IMG_SIZE, 3], name='cer_ph')
cer_img_tensor = cer_ph / MAX_PIXEL_VALUE
ema = tf.train.ExponentialMovingAverage(decay=0.9)
ave = tf.reduce_mean(cer_img_tensor, axis=[0,1,2])
with tf.control_dependencies([ema.apply([ave])]):
    channel_averages = ema.average(ave)
cer_img_tensor = cer_img_tensor - channel_averages

with tf.name_scope('generator'):
    with tf.variable_scope('generator'):
        g_mnist = generator(cer_img_tensor,'generator')

# 读入mnist
mnist_ph = tf.placeholder(tf.float32, shape=[BATCH_SIZE, MNIST_IMG_SIZE, MNIST_IMG_SIZE], name='mnist_ph')
mnist_img_tensor = mnist_ph / MAX_PIXEL_VALUE
mnist_img_tensor = tf.expand_dims(mnist_img_tensor,axis=3)

# 得到fw的评分
with tf.name_scope('critic_fake'):
    with tf.variable_scope('critic'):
        fw_g_mnist = critic(g_mnist)
with tf.name_scope('critic_real'):
    with tf.variable_scope('critic', reuse=True):
        fw_mnist = critic(mnist_img_tensor)

# 梯度下降
optm_c, optm_g = get_optimizers(fw_g_mnist, fw_mnist, 0.001)

# 可视化
sum_g_mnist = tf.summary.image('g_mnist', g_mnist, max_outputs=BATCH_SIZE)
sum_image_fake = tf.summary.histogram('image_fake', g_mnist)
sum_image_real = tf.summary.histogram('image_real', mnist_img_tensor)

sum_fw_g_mnist = tf.summary.scalar('fw_g_mnist', tf.reduce_mean(fw_g_mnist))
sum_fw_mnist = tf.summary.scalar('fw_mnist', tf.reduce_mean(fw_mnist))

merge_all = tf.summary.merge_all()

saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
global_step = 1

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    file_writer = tf.summary.FileWriter(EVENT_DIR, graph=sess.graph)

    while 1:
        if global_step <= 100:
            n_critic = N_CRITIC_1
            write_summary_step = WRITE_SUMMARY_STEP_1
            save_step = SAVE_STEP_1
        else:
            n_critic = N_CRITIC_2
            write_summary_step = WRITE_SUMMARY_STEP_2
            save_step = SAVE_STEP_2
        # n_critic = 1
        flag_write_summary = True if global_step % write_summary_step== 0 else False
        for i in range(n_critic):
            if (i+1)%10==0:
                print('critics: ' + str(i))
            cer_batch = cer_batch_mngr.get_batch()
            mnist_batch = mnist_batch_mngr.get_batch()
            if flag_write_summary and i == n_critic - 1:
                _, merge = sess.run([optm_c, merge_all], feed_dict={cer_ph: cer_batch, mnist_ph: mnist_batch})
                file_writer.add_summary(merge, global_step=global_step)
                file_writer.flush()
            else:
                sess.run(optm_c, feed_dict={cer_ph: cer_batch, mnist_ph: mnist_batch})

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

        if global_step % save_step == 0:
            saver.save(sess, SAVE_PATH+'/4_11.cpkt', global_step=global_step)
            print('Saved! Step: '+ str(global_step))

        global_step += 1
