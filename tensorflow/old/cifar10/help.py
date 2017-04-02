from PIL import Image

import tensorflow as tf

import cifar10


width = 24
height = 24

categories = [ "zero","um", "dois" ]

filename = "/home/marcos/Documentos/Deep_Learning/evaluate/2.jpg" # absolute path to input image

im = Image.open(filename)





im.save(filename, format='JPEG', subsampling=0, quality=100)

input_img = tf.image.decode_jpeg(tf.read_file(filename), channels=3)
tf_cast = tf.cast(input_img, tf.float32)

float_image = tf.image.resize_image_with_crop_or_pad(tf_cast, height, width)

images = tf.expand_dims(float_image, 0)

logits = cifar10.inference(images)

_, top_k_pred = tf.nn.top_k(logits, k=3)

init_op = tf.initialize_all_variables()

# with tf.Session() as sess:


#   sess.run(init_op)
#   _, top_indices = sess.run([_, top_k_pred])
#   for key, value in enumerate(top_indices[0]):
#     print (categories[value] + ", " + str(_[0][key]))



# # 1. GRAPH CREATION
# input_img = tf.image.decode_jpeg(tf.read_file("/home/marcos/Documentos/Deep_Learning/evaluate/2.jpg"), channels=3)
# reshaped_image = tf.image.resize_image_with_crop_or_pad(tf.cast(input_img, width, height), tf.float32)
# float_image = tf.image.per_image_withening(reshaped_image)
# images = tf.expand_dims(float_image, 0)  # create a fake batch of images (batch_size = 1)
# logits = faultnet.inference(images)
# _, top_k_pred = tf.nn.top_k(logits, k=3)

# 2. TENSORFLOW SESSION
with tf.Session() as sess:
  sess.run(init_op)

  top_indices = sess.run([top_k_pred])
  print ("Predicted ", top_indices[0], " for your input image.")  