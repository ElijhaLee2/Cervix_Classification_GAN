import tensorflow as tf

scope = tf.name_scope('scope1')

with scope:
    qqq = tf.summary.scalar('sum1', tf.convert_to_tensor([1]))

with scope:
    tf.summary.scalar('sum2', tf.convert_to_tensor([2]))

sum_coll = tf.get_collection(tf.GraphKeys.SUMMARIES,scope=scope)

print()