import model
import load_data
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

class make_dictionary():
    def __init__(self, epoch):
        self.epoch = epoch
        self.PDSN = model.PDSN_model()
        self.PDSN.build((1,28,28,2))
        self.PDSN.summary()
        self.optimizer = Adam(learning_rate=1e-4, decay=1e-7)
        self.checkpoint = tf.train.Checkpoint(model=self.PDSN, optimizer=self.optimizer)

        checkpoint_root = 'model_checkpoint/'
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_root))


    @tf.function
    def get_Z_different(self,Nonocc_image, Occ_image):
        inputs = tf.concat([Nonocc_image, Occ_image], axis=-1)
        Z_different, _, _, _ = self.PDSN(inputs)
        return Z_different

    def binary_mask(self,m,persent=0.45):
        mask_ = tf.compat.v1.layers.flatten(m)
        mask_copy = tf.identity(mask_)
        sorted_mask = tf.sort(mask_copy, axis=0)
        value = sorted_mask[:, int(m.shape[1] * m.shape[2] * m.shape[3] * persent)]
        value = tf.reshape(value, shape=(tf.shape(m)[0], 1, 1, 1))
        bool = tf.math.greater(m, value)
        binary_mask = tf.where(bool, 0, 1)
        binary_mask = tf.cast(binary_mask, tf.float32)
        return binary_mask

    def min_max_normalizing(self,Z_different):
        Z_diiferent_max, Z_different_min = tf.reduce_max(Z_different,axis=(1,2,3)), tf.reduce_min(Z_different,axis=(1,2,3))
        Z_different = tf.math.divide(Z_different - Z_different_min, Z_diiferent_max - Z_different_min)
        return Z_different

    def make_M(self, do_binary=True):
        '''
        Generator jth block occ. region for mean Z_different.
        all 25 blocks mean Z_different ==> Z_different dictionary
        :return: jth block Z_different
        '''
        x_train, x_test, y_train, y_test = load_data.load_mnist()
        train_occ, train_nonocc, train_label = load_data.prepare_data(x_train, y_train)
        Z_differents = self.get_Z_different(train_nonocc, train_occ) #(60000, 8, 8, 512)
        Z_differents_normalizing = self.min_max_normalizing(Z_differents)
        Z_differents_mean = tf.reduce_mean(Z_differents_normalizing,axis=0)
        if do_binary:
            Z_differents_mean = self.binary_mask(Z_differents_mean)
        return Z_differents_mean
