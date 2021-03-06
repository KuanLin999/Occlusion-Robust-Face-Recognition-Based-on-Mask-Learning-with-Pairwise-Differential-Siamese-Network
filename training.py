import model
import load_data
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam,SGD
import datetime
import losses
import time
from sklearn.metrics import accuracy_score

class training_PDSN():
    def __init__(self, epoch, batch_size):
        self.epoch = epoch
        self.batch_size = batch_size
        self.PDSN = model.PDSN_model()
        self.PDSN.build((1,28,28,2))
        self.PDSN.summary()
        self.optimizer = Adam(learning_rate=1e-4, decay=1e-7)
        self.checkpoint = tf.train.Checkpoint(model=self.PDSN, optimizer=self.optimizer)

    @tf.function
    def train_step(self,Nonocc_image, Occ_image, label):
        inputs = tf.concat([Nonocc_image, Occ_image], axis=-1)
        label = tf.cast(label, tf.float32)
        with tf.GradientTape() as tape:
            _,Nonocc_feature_x_Z_different, Occ_feature_x_Z_different, pred_label_for_Occ = self.PDSN(inputs)
            different_loss = losses.different_loss(Nonocc_feature_x_Z_different, Occ_feature_x_Z_different)
            CE_loss_for_Occ = losses.cls_loss(label, pred_label_for_Occ)
            total_loss = different_loss + CE_loss_for_Occ
        grads = tape.gradient(total_loss, self.PDSN.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.PDSN.trainable_variables))

    @tf.function
    def test_step(self, Nonocc_image, Occ_image, label):
        inputs = tf.concat([Nonocc_image, Occ_image], axis=-1)
        label = tf.cast(label, tf.float32)
        _, Nonocc_feature_x_Z_different, Occ_feature_x_Z_different, pred_label_for_Occ = self.PDSN(inputs)
        different_loss = losses.different_loss(Nonocc_feature_x_Z_different, Occ_feature_x_Z_different)
        CE_loss_for_Occ = losses.cls_loss(label, pred_label_for_Occ)
        total_loss = different_loss + CE_loss_for_Occ
        return different_loss,CE_loss_for_Occ,total_loss,pred_label_for_Occ

    def plot_training_stage(self, train_writer, val_writer, train_data, val_data, loss_name, e, train_description=None, val_description=None):
        with train_writer.as_default():
            tf.summary.scalar(loss_name, train_data, step=e + 1, description=train_description)
        with val_writer.as_default():
            tf.summary.scalar(loss_name, val_data, step=e + 1, description=val_description)

    def caluate_loss(self, occ, nonocc, label):
        different_loss,CE_loss_for_Occ,total_loss,pred_label_for_Occ = self.test_step(nonocc, occ, label)
        pred_label_for_Occ = tf.argmax(pred_label_for_Occ,axis=-1)
        label = np.argmax(label,axis=-1)
        accuracy = accuracy_score(pred_label_for_Occ,label)
        return different_loss.numpy(), CE_loss_for_Occ.numpy(), total_loss.numpy(), accuracy

    def training(self):
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        path = 'model_checkpoint/training_first_block/tb_logs_'
        train_log_dir = path + current_time + '/train'
        val_log_dir = path + current_time + '/val'
        train_writer = tf.summary.create_file_writer(train_log_dir)
        val_writer = tf.summary.create_file_writer(val_log_dir)
        checkpoint_dir = f'model_checkpoint/training_first_block/training_checkpoints_{current_time}/model'
        x_train, x_test, y_train, y_test = load_data.load_mnist()
        min_total_loss = 0
        plot_curve_name = ['different loss(epoch)', 'cls loss(epoch)', 'total loss(epoch)', 'accuracy(epoch)']
        for epoch in range(self.epoch):
            new_epoch_start_time = time.time()
            train_occ, train_nonocc, train_label = load_data.prepare_data(x_train,y_train)
            test_occ, test_nonocc, test_label = load_data.prepare_data(x_test,y_test)
            if epoch != 0:
                for batch in range(int(train_nonocc.shape[0]/self.batch_size)):
                    nonocc, occ, labels = load_data.batch_data(train_nonocc,train_occ,train_label,batch,self.batch_size)
                    self.train_step(nonocc, occ, labels)

            train_L_diff, train_L_cls, train_L_total, train_accuracy = [],[],[],[]
            for batch in range(int(train_nonocc.shape[0] / self.batch_size)):
                nonocc, occ, labels = load_data.batch_data(train_nonocc, train_occ, train_label, batch, self.batch_size)
                L_diff, L_cls, L_total, accuracy = self.caluate_loss(nonocc, occ, labels)
                train_L_diff.append(L_diff), train_L_cls.append(L_cls), train_L_total.append(L_total), train_accuracy.append(accuracy)

            test_L_diff, test_L_cls, test_L_total, test_accuracy = [],[],[],[]
            for batch in range(int(test_nonocc.shape[0] / self.batch_size)):
                nonocc, occ, labels = load_data.batch_data(test_nonocc, test_occ, test_label, batch, self.batch_size)
                L_diff, L_cls, L_total, accuracy = self.caluate_loss(nonocc, occ, labels)
                test_L_diff.append(L_diff), test_L_cls.append(L_cls), test_L_total.append(L_total), test_accuracy.append(accuracy)

            trains = [train_L_diff, train_L_cls, train_L_total, train_accuracy]
            tests = [test_L_diff, test_L_cls, test_L_total, test_accuracy]

            end_now_epoch_time = time.time()
            print('==============', 'epoch:', epoch, '==============')
            print('spant time:', end_now_epoch_time - new_epoch_start_time)
            for idx in range(len(plot_curve_name)):
                name = plot_curve_name[idx]
                train, test = trains[idx], tests[idx]
                self.plot_training_stage(train_writer, val_writer, np.mean(train), np.mean(test), name, epoch)
                print('train %s:' % name, np.mean(train), 'test %s:' % name, np.mean(test))

            if epoch==0:
                min_total_loss = np.mean(test_L_total)

            elif np.mean(test_L_total) < min_total_loss:
                self.checkpoint.save(checkpoint_dir)
                min_total_loss = np.mean(test_L_total)
                print('model save.')

gogogo = training_PDSN(epoch=100,batch_size=64)
gogogo.training()