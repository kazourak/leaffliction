import tensorflow as tf


def init_tf_env():
    """
    Set up the tensorflow environment.
    Returns
    -------

    """
    tf.keras.backend.clear_session()

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3096)]
            )
            print("Memory growth enabled for GPUs.")
