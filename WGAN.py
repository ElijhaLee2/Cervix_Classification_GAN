from GAN_models import *
from tensorflow.examples.tutorials.mnist import input_data
from HyperParam import *
import os

os.popen('killall tensorboard')
file_list = os.listdir(EVENT_DIR)
for file in file_list:
    os.remove(os.path.join(EVENT_DIR, file))

# 将数据集读入内存

# 生成假图片
z_ph = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 3584, 3584, 3], name='z_ph')
with tf.variable_scope('generator'):
    g_z = generator(z_ph)

# 读入真图片
x_ph = tf.placeholder(tf.float32, shape=[batch_size, mnist_height, mnist_width, 1], name='x_ph')

# 得到fw的评分
with tf.variable_scope('critic'):
    fw_g_z = critic(g_z, name='critic_g_z')
with tf.variable_scope('critic', reuse=True):
    fw_x = critic(x_ph, name='critic_x')

# # 梯度下降laoshi
# loss_c, loss_g, \
# vars_c, vars_g, \
# grad_c, grad_g, \
# optm_c, optm_g \
#     = get_optimizers(fw_g_z, fw_x, learning_rate)
#

# xin tiduxiajiang
optm_c, optm_g = get_optimizers(fw_g_z, fw_x, learning_rate)

# 可视化
sum_g_z = tf.summary.image('g_z', g_z, max_outputs=batch_size)
sum_image_fake = tf.summary.histogram('image_fake',g_z)
sum_image_real = tf.summary.histogram('image_real', x_ph)

sum_fw_g_z = tf.summary.scalar('fw_g_z', tf.reduce_mean(fw_g_z))
sum_fw_x = tf.summary.scalar('fw_x', tf.reduce_mean(fw_x))

# sum_loss_c = tf.summary.scalar('loss_c', loss_c)
# sum_loss_g = tf.summary.scalar('loss_g', loss_g)
#
# sum_grad_c = [tf.summary.histogram('grad_'+g[1].name, g[0]) for g in grad_c]
# sum_grad_g = [tf.summary.histogram('grad_'+g[1].name, g[0]) for g in grad_g]
#
# sum_vars_c = [tf.summary.histogram(v.name, v) for v in vars_c]
# sum_vars_g = [tf.summary.histogram(v.name, v) for v in vars_g]

merge_all = tf.summary.merge_all()

saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
global_step = 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    file_writer = tf.summary.FileWriter(event_path1, graph=sess.graph)

    while 1:
        n_critic = n_critic_1 if global_step <= 100 else n_critic_2
        flag_write_summary = True if global_step % write_summary_step == 0 else False
        # flag_write_summary = False
        for i in range(n_critic):
            z = get_z()
            x, _ = mnist.train.next_batch(batch_size)
            x = np.reshape(x, [batch_size, mnist_height, mnist_width, 1])
            # if i!=n_critic-1 and (not flag_write_summary):
            #     sess.run(optm_c, feed_dict={z_ph: z, x_ph: x})
            # elif i!=n_critic-1 and flag_write_summary:
            #     _, merge = sess.run([optm_c, merge_all], feed_dict={z_ph: z, x_ph: x})
            #     file_writer.add_summary(merge, global_step=global_step)
            #     file_writer.flush()
            # elif i == n_critic - 1 and not flag_write_summary:
            #     sess.run([optm_c, optm_g], feed_dict={z_ph: z, x_ph: x})
            # elif i == n_critic - 1 and flag_write_summary:
            #     _, _, merge = sess.run([optm_c, optm_g, merge_all], feed_dict={z_ph: z, x_ph: x})
            #     file_writer.add_summary(merge, global_step=global_step)
            #     file_writer.flush()
            # else:
            #     print('EXM?!')

            if flag_write_summary and i == n_critic - 1:
                _, merge = sess.run([optm_c, merge_all], feed_dict={z_ph: z, x_ph: x})
                file_writer.add_summary(merge, global_step=global_step)
                file_writer.flush()
            else:
                sess.run(optm_c, feed_dict={z_ph: z, x_ph: x})

        if flag_write_summary:
            _, merge = sess.run([optm_g, merge_all], feed_dict={z_ph: z, x_ph: x})
            file_writer.add_summary(merge, global_step=global_step)
            file_writer.flush()
        else:
            sess.run(optm_g, feed_dict={z_ph: z, x_ph: x})

        if global_step % display_step == 0:
            print("Step:" + str(global_step))

        if global_step % save_step == 0:
            saver.save(sess, save_path1, global_step=global_step)

        global_step += 1


# time.sleep(5)
# os.popen('tensorboard --logdir=/home/elijha/PycharmProjects/WGAN/Events/3_12')
