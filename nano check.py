import tensorflow as tf
import time

print("GPUs:", tf.config.list_physical_devices('GPU'))

# Enable memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# Large matrix multiplication
with tf.device('/GPU:0'):
    a = tf.random.normal([8000, 8000])
    b = tf.random.normal([8000, 8000])

    start = time.time()
    c = tf.matmul(a, b)
    tf.reduce_sum(c).numpy()
    end = time.time()

print("Time taken:", end - start)