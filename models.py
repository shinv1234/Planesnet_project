import numpy as np
import tensorflow as tf
import math


class Plain_Model():
    def __init__(self, sess, model_name, learning_rate):
        self.sess = sess
        self.model_name = model_name
        self.learning_rate = learning_rate
        self._build_net(self.learning_rate)
        
    def _build_net(self, learning_rate):
        with tf.variable_scope('{}_data'.format(self.model_name)) as scope:
            self.X = tf.placeholder(tf.float32, [None, 20, 20, 3])
            self.Y = tf.placeholder(tf.float32, [None, 1])
            self.learning_rate = learning_rate
            
        with tf.variable_scope('{}_conv1'.format(self.model_name)) as scope:
            W1 = tf.Variable(tf.random_normal([4, 4, 3, 32], stddev=0.01), name='weight1')
            L1 = tf.nn.conv2d(self.X, W1, strides=[1, 1, 1, 1], padding='SAME')
            L1 = tf.nn.relu(L1)
            L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            L1 = tf.reshape(L1, [-1, 10 * 10 * 32])
            
        with tf.name_scope('{}_fully_connected_layer1'.format(self.model_name)) as scope:
            W2 = tf.get_variable('W2', shape=[10 * 10 * 32, 1], initializer=tf.contrib.layers.xavier_initializer())        
            b = tf.Variable(tf.random_normal([1]))
            self.hypothesis = tf.sigmoid(tf.matmul(L1, W2) + b)
            
        with tf.name_scope('{}_cost'.format(self.model_name)) as scope:
            self.cost = tf.losses.sigmoid_cross_entropy(logits=hypothesis, multi_class_labels=Y)
            # self.cost = -tf.reduce_mean(self.Y * tf.log(self.hypothesis + 0.01) + (1 - self.Y) * tf.log(1 - self.hypothesis + 0.01))
            cost_summary = tf.summary.scalar('cost', self.cost)
            
        with tf.name_scope('{}_train_optimizer'.format(self.model_name)) as scope:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)  
        
        self.predicted = tf.cast(self.hypothesis > 0.5, dtype=tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predicted, self.Y), dtype=tf.float32))
        accuracy_summary = tf.summary.scalar('{}_accuracy'.format(self.model_name), self.accuracy)
        
    def summary(self, summary_directory):
        self.summary_directory = summary_directory # ex: './logs/planesnet2_log'
        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.summary_directory)
        self.writer.add_graph(self.sess.graph)
        
    def train(self, x_data, y_data):
        return self.sess.run([self.cost, self.optimizer, self.merged_summary], feed_dict={self.X: x_data, self.Y: y_data})
    
    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test})
        
    def predict(self, x_test):
        return self.sess.run(self.hypothesis, feed_dict={self.X: x_test})
    

class Resnet_Model(Plain_Model):
    def __init__(self, sess, model_name, learning_rate):
        self.sess = sess
        self.model_name = model_name
        self.learning_rate = learning_rate
        self._build_net(self.learning_rate)
        
    def _build_net(self, learning_rate):
        with tf.variable_scope('{}_data'.format(self.model_name)) as scope:
            self.X = tf.placeholder(tf.float32, [None, 20, 20, 3])
            self.Y = tf.placeholder(tf.float32, [None, 1])
            self.learning_rate = learning_rate
            
        with tf.variable_scope('{}_conv1'.format(self.model_name)) as scope:
            CL_W1 = tf.Variable(tf.random_normal([7, 7, 3, 16], stddev=0.01), name='conv1_weight1')
            CL1 = tf.nn.conv2d(self.X, CL_W1, strides=[1, 2, 2, 1], padding='SAME', name='conv1_conv')
            CL1_bn_mean1, CL1_bn_var1 = tf.nn.moments(CL1, [0])
            CL1_bn_scale1 = tf.Variable(tf.ones([10, 10, 16]))
            CL1_bn_beta1 = tf.Variable(tf.zeros([10, 10, 16]))
            CL1_bn_epsilon1 = 0.001
            CL1 = tf.nn.batch_normalization(CL1, CL1_bn_mean1, 
                                            CL1_bn_var1, CL1_bn_beta1, 
                                            CL1_bn_scale1, CL1_bn_epsilon1, name='conv1_batch1')
            CL1 = tf.nn.relu(CL1)
            CL1 = tf.nn.max_pool(CL1, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
                
        with tf.variable_scope('{}_residual_1'.format(self.model_name)) as scope:
            RW1_1 = tf.Variable(tf.random_normal([3, 3, 16, 16], stddev=0.01), name='residual_1_weight1')
            RL1 = tf.nn.conv2d(CL1, RW1_1, strides=[1, 1, 1, 1], padding='SAME', name='residual_1_conv1')
            R1_1_bn_mean1, R1_1_bn_var1 = tf.nn.moments(RL1, [0])
            R1_1_bn_scale1 = tf.Variable(tf.ones([10, 10, 16]))
            R1_1_bn_beta1 = tf.Variable(tf.zeros([10, 10, 16]))
            R1_1_bn_epsilon1 = 0.001
            RL1 = tf.nn.batch_normalization(RL1, R1_1_bn_mean1, R1_1_bn_var1, R1_1_bn_beta1, 
                                            R1_1_bn_scale1, R1_1_bn_epsilon1, name='Residual_1_batch1')
            RL1 = tf.nn.relu(RL1, name='residual_1_relu1')
            
            RW1_2 = tf.Variable(tf.random_normal([3, 3, 16, 16], stddev=0.01), name='residual_1_weight2') 
            RL1 = tf.nn.conv2d(RL1, RW1_2, strides=[1, 1, 1, 1], padding='SAME', name='residual_1_conv2')
            R1_2_bn_mean1, R1_2_bn_var1 = tf.nn.moments(RL1, [0])
            R1_2_bn_scale1 = tf.Variable(tf.ones([10, 10, 16]))
            R1_2_bn_beta1 = tf.Variable(tf.zeros([10, 10, 16]))
            R1_2_bn_epsilon1 = 0.001
            RL1 = tf.nn.batch_normalization(RL1, R1_2_bn_mean1, R1_2_bn_var1, R1_2_bn_beta1, 
                                            R1_2_bn_scale1, R1_2_bn_epsilon1, name='Residual_1_batch2')
            RL1 = tf.nn.relu(RL1, name='residual_1_relu1')
            
            RL1 = RL1 + CL1
            
        with tf.variable_scope('{}_residual_2'.format(self.model_name)) as scope:
            RW2_1 = tf.Variable(tf.random_normal([3, 3, 16, 32], stddev=0.01), name='residual_2_weight1') 
            RL2 = tf.nn.conv2d(RL1, RW2_1, strides=[1, 1, 1, 1], padding='SAME', name='residual_2_conv1')
            R2_1_bn_mean1, R2_1_bn_var1 = tf.nn.moments(RL2, [0])
            R2_1_bn_scale1 = tf.Variable(tf.ones([10, 10, 32]))
            R2_1_bn_beta1 = tf.Variable(tf.zeros([10, 10, 32]))
            R2_1_bn_epsilon1 = 0.001
            RL2 = tf.nn.batch_normalization(RL2, R2_1_bn_mean1, R2_1_bn_var1, R2_1_bn_beta1, 
                                            R2_1_bn_scale1, R2_1_bn_epsilon1, name='Residual_2_batch1')
            RL2 = tf.nn.relu(RL2)
            RW2_2 = tf.Variable(tf.random_normal([3, 3, 32, 32], stddev=0.01), name='residual_2_weight2') 
            RL2 = tf.nn.conv2d(RL2, RW2_2, strides=[1, 1, 1, 1], padding='SAME', name='residual_2_conv2')
            R2_2_bn_mean1, R2_2_bn_var1 = tf.nn.moments(RL2, [0])
            R2_2_bn_scale1 = tf.Variable(tf.ones([10, 10, 32]))
            R2_2_bn_beta1 = tf.Variable(tf.zeros([10, 10, 32]))
            R2_2_bn_epsilon1 = 0.001
            RL2 = tf.nn.batch_normalization(RL2, R2_2_bn_mean1, R2_2_bn_var1, R2_2_bn_beta1, 
                                            R2_2_bn_scale1, R2_2_bn_epsilon1, name='Residual_2_batch2')
            RL2 = tf.nn.relu(RL2)
            
            R2_SC_W = tf.Variable(tf.random_normal([3, 3, 16, 32], stddev=0.01), name='residual_2_shortcut_weight1') 
            R_SC_L1 = tf.nn.conv2d(RL1, R2_SC_W, strides=[1, 1, 1, 1], padding='SAME', name='residual_2_shortcut_conv1')
            
            RL2 = RL2 + R_SC_L1
            
        with tf.variable_scope('{}_reshape'.format(self.model_name)) as scope:
            FCL1 = tf.reshape(RL2, [-1, 10 * 10 * 32])
            
        with tf.variable_scope('{}_fully_connected_layer1'.format(self.model_name)) as scope:
            FCL_W1 = tf.get_variable('FCL_W1', shape=[10 * 10 * 32, 1], initializer=tf.contrib.layers.xavier_initializer()) 
            FCL_b1 = tf.Variable(tf.random_normal([1]))
            self.hypothesis = tf.sigmoid(tf.matmul(FCL1, FCL_W1) + FCL_b1)
            
        with tf.name_scope('{}_cost'.format(self.model_name)) as scope:
            self.cost = -tf.reduce_mean(self.Y * tf.log(self.hypothesis + 0.00001) + (1 - self.Y) * tf.log(1 - self.hypothesis + 0.00001))
            cost_summary = tf.summary.scalar('cost', self.cost)
            
        with tf.name_scope('{}_train_optimizer'.format(self.model_name)) as scope:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)  
            
        self.predicted = tf.cast(self.hypothesis > 0.5, dtype=tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predicted, self.Y), dtype=tf.float32))
        accuracy_summary = tf.summary.scalar('{}_accuracy'.format(self.model_name), self.accuracy)
        

class Model_Training():
    def __init__(self, model_instance, train_data, train_labels, test_data, test_labels):
        self.model_instance = model_instance
        self.sess = model_instance.sess
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        
        
    def shuffle_batch(self, batch_size, capacity, min_after_dequeue, seed=10):
        self.batch_size = batch_size
        self.capacity = capacity
        self.min_after_dequeue = min_after_dequeue
        self.seed = seed
        self.train_data_batch, self.train_labels_batch = tf.train.shuffle_batch([self.train_data, self.train_labels], 
                                                              capacity=self.capacity, min_after_dequeue=self.min_after_dequeue, 
                                                              enqueue_many=True , batch_size=self.batch_size, seed=self.seed,
                                                              allow_smaller_final_batch=True)
        
        
    def model_saver(self, directory, saver):
        self.saver = saver # tf.train.Saver()
        self.save_directory = directory # "./model_saver/planenet3_model_saver/planenet3_model_save"
        
    def model_restore(self, directory, saver):
        self.restore = saver # tf.train.Saver()
        self.restore_directory = directory # "./model_saver/planenet3_model_saver/planenet3_model_save"
    
    def train_model(self, epoch, init=tf.global_variables_initializer()):
        self.init = init
        self.epoch = epoch
        self.batch_steps = math.ceil(len(self.train_data) / self.batch_size)
        
        self.sess.run(self.init)
        
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        
        for epoch_step in range(epoch):
            total_cost = 0
            for batch_step in range(self.batch_steps):
                x_batch, y_batch = self.sess.run([self.train_data_batch, self.train_labels_batch])
                cost_val, _, summary = plain_model.train(x_data=x_batch, y_data=y_batch)
                self.model_instance.writer.add_summary(summary, global_step=epoch_step)
                total_cost += cost_val
            self.cost = total_cost / self.batch_size
            print('Epoch:', '%04d' % (epoch_step + 1),
                  'Avg. cost =', '{:.5f}'.format(self.cost), 
                  'Train Accuracy: ', '{:.6f}'.format(self.model_instance.get_accuracy(x_test=self.train_data, y_test=self.train_labels)),
                  'Test Accuracy: ', '{:.6f}'.format(self.model_instance.get_accuracy(x_test=self.test_data, y_test=self.test_labels)))
            
        self.coord.request_stop()
        self.coord.join(self.threads)
        save_path = self.saver.save(self.sess, self.save_directory)
        print("Model saved in file: %s" % save_path)
        
    def retrain_model(self, epoch, init=tf.global_variables_initializer()):
        self.init = init
        self.epoch = epoch
        self.batch_steps = math.ceil(len(self.train_data) / self.batch_size)
        
        self.sess.run(self.init)
        self.restore.restore(self.sess, self.restore_directory)

        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        
        for epoch_step in range(epoch):
            total_cost = 0
            for batch_step in range(self.batch_steps):
                x_batch, y_batch = self.sess.run([self.train_data_batch, self.train_labels_batch])
                cost_val, _, summary = plain_model.train(x_data=x_batch, y_data=y_batch)
                self.model_instance.writer.add_summary(summary, global_step=epoch_step)
                total_cost += cost_val
            self.cost = total_cost / self.batch_size
            print('Epoch:', '%04d' % (epoch_step + 1),
                  'Avg. cost =', '{:.5f}'.format(self.cost), 
                  'Train Accuracy: ', '{:.6f}'.format(self.model_instance.get_accuracy(x_test=self.train_data, y_test=self.train_labels)),
                  'Test Accuracy: ', '{:.6f}'.format(self.model_instance.get_accuracy(x_test=self.test_data, y_test=self.test_labels)))
            
        self.coord.request_stop()
        self.coord.join(self.threads)
        save_path = self.saver.save(self.sess, self.save_directory)
        print("Model saved in file: %s" % save_path)
