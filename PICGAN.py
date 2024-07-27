import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from pickle import load
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.layers import LSTM, Dense, Flatten, Conv1D, BatchNormalization, LeakyReLU, ELU, ReLU, RepeatVector, TimeDistributed, Reshape
from tensorflow.keras import Sequential, regularizers
from tensorflow.python.client import device_lib
from keras.models import model_from_json
from keras.utils.vis_utils import plot_model
import gc

# Load data
X_train = np.load("X_train.npy", allow_pickle=True)
y_train = np.load("y_train.npy", allow_pickle=True)
yc_train = np.load("yc_train.npy", allow_pickle=True)

X_train_IDM = np.load("X_train_IDM.npy", allow_pickle=True)
y_train_IDM = np.load("y_train_IDM.npy", allow_pickle=True)
yc_train_IDM = np.load("yc_train_IDM.npy", allow_pickle=True)
print(X_train.shape, y_train.shape, yc_train.shape)
print(X_train_IDM.shape, y_train_IDM.shape, yc_train_IDM.shape)
#load data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')
test_pairs = pd.read_csv('test_pairs.csv')
#数据标准化
train_data = train_data.drop(['Vehicle_ID','Leader_ID','Frame_ID','LocalY','LocalY_leader',
                              'Mean_Speed_leader','Mean_Acceleration_leader','Vehicle_length',
                              'Vehicle_Class_ID','Lane_ID','Follower_ID','Vehicle_length_leader'],axis=1)
train_data = train_data.values
mean = train_data[:].mean(axis=0)
std = train_data[:].std(axis=0)

test_data = test_data.values
test_data[:,[4,5,6,7]] -= mean
test_data[:,[4,5,6,7]] /= std

test_data = pd.DataFrame(test_data)
test_pairs = test_pairs.values
# Define the generator
def Generator(input_dim, output_dim, feature_size_in, feature_size_out, img_shape) -> tf.keras.models.Model:
    model = Sequential()
    model.add(LSTM(32, activation='tanh', input_shape=(input_dim, feature_size_in)))
    model.add(RepeatVector(output_dim))
    model.add(LSTM(32, activation='tanh', return_sequences=True))
    model.add(TimeDistributed(Dense(feature_size_out)))
    model.add(Reshape(img_shape))
    return model
# Define the discriminator
def Discriminator_c(xy_shape) -> tf.keras.models.Model:
    model = tf.keras.Sequential()
    model.add(Flatten(input_shape=xy_shape))
    model.add(Dense(64))
    model.add(Dense(64))
    #model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1, activation='sigmoid'))
    return model
# Define the discriminator
def Discriminator_p(xy_shape) -> tf.keras.models.Model:
    model = tf.keras.Sequential()
    model.add(Flatten(input_shape=xy_shape))
    model.add(Dense(64))
    model.add(Dense(64))
    #model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1, activation='sigmoid'))
    return model
class GAN():
    def __init__(self, generator, discriminator_c, discriminator_p, opt):
        self.opt = opt
        self.lr = opt["lr"]
        self.generator = generator
        self.discriminator_c = discriminator_c
        self.discriminator_p = discriminator_p
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.discriminator_c_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.discriminator_p_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.batch_size = self.opt['bs']
        self.checkpoint_dir = '../training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_c_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator_c)
        
        model_json = generator.to_json()
        with open("generator_model.json", "w") as json_file:
            json_file.write(model_json)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def sample_images(self, generator, epoch):
        
        plt.figure(figsize=(20, 4))
        m = np.random.randint(0, len(test_pairs), 5)

        for _ in np.arange(len(m)):
            p, q = test_pairs[m[_]]
            eachtestpair = pd.DataFrame(test_data[(test_data[0] == p) & (test_data[1] == q)]).reset_index(drop=True)
            eachtestpair = eachtestpair.values

            lookback, delay = input_dim, output_dim
            dt = 0.1
            pred = list()
            for i in range(lookback):
                eachtimestep = eachtestpair[i]
                eachtimestep = eachtimestep.tolist()
                pred.append(eachtimestep)
            pred = np.array(pred)
            j = pred.shape[0]
            while j < len(eachtestpair):
                test_x = pred[(j - lookback):j, [4,5,6,7]]
                test_x = test_x.reshape(1,test_x.shape[0],test_x.shape[1])
                yhat = generator.predict(test_x,verbose=0)
                yhat = yhat.reshape(yhat.shape[0]*yhat.shape[1],yhat.shape[2])
                yhat[:,0] = yhat[:,0] * std[3] + mean[3]
                pred = pred.tolist()
                n = j + delay
                if n > len(eachtestpair):
                    n = len(eachtestpair)
                for k in range(j, n):        
                    eachtimestep_pred = eachtestpair[k]
                    eachtimestep_pred = eachtimestep_pred.tolist()
                    eachtimestep_pred[7] = yhat[k-j,0]
                    eachtimestep_pred[6] = (pred[-1][6] * std[2] + mean[2]) + (yhat[k-j,0] * dt)
                    if eachtimestep_pred[6] < 0 :
                        eachtimestep_pred[6] = 0
                        eachtimestep_pred[7] = (eachtimestep_pred[6] - (pred[-1][6] * std[2] + mean[2]))/dt
                    eachtimestep_pred[5] = eachtimestep_pred[9] - eachtimestep_pred[6]
                    eachtimestep_pred[3] = pred[-1][3] + 0.5 * (pred[-1][6] * std[2] + mean[2] + eachtimestep_pred[6]) * dt
                    eachtimestep_pred[4] = eachtimestep_pred[8] - eachtimestep_pred[3] - eachtimestep_pred[15]

                    eachtimestep_pred[4:8] -= mean
                    eachtimestep_pred[4:8] /= std
                    pred.append(eachtimestep_pred)
                pred = np.array(pred)
                j = pred.shape[0]
            pred_each = pd.DataFrame(pred)

            eachtestpair = pd.DataFrame(eachtestpair)
            pred_each.loc[:,[4,5,6,7]] = pred_each.loc[:,[4,5,6,7]] * std + mean
            eachtestpair.loc[:,[4,5,6,7]] = eachtestpair.loc[:,[4,5,6,7]] * std + mean

            plt.subplot(1, 5, _+1)
            plt.plot(eachtestpair[2],eachtestpair[8],color='blue',linewidth=1,linestyle='--',label='obs')
            plt.plot(eachtestpair[2],eachtestpair[3],color='blue',linewidth=1,linestyle='--',label='obs')
            plt.plot(pred_each[2],pred_each[3],color='green',linewidth=1,linestyle='-',label='sim')
            plt.legend(loc='upper right', frameon=True)
            plt.title('%d,%d' %(p,q),fontsize=12)

        #plt.xlabel('Time[s]',fontsize=12)
        #plt.ylabel('LocalY[m]',fontsize=12)
        plt.savefig('plots/%d.png' % (epoch+1))
        # close the figure to free up memory
        plt.close()

        # Set testpair and pred_each to None
        eachtestpair = None
        pred_each = None
        pred = None

        gc.collect()

    #@tf.function
    def train_step(self, data, data_IDM, idx_0):
        # Get a random batch of real images
        idx = np.arange(idx_0, idx_0+self.batch_size, 1)
        X_train, y_train, yc_train = data
        real_input = X_train[idx]
        real_y = y_train[idx]
        yc = yc_train[idx]
        X_train_IDM, y_train_IDM, yc_train_IDM = data_IDM
        real_input_IDM = X_train_IDM[idx]
        real_y_IDM = y_train_IDM[idx]
        yc_IDM = yc_train_IDM[idx]

        # Train the discriminator
        with tf.GradientTape() as disc_c_tape:
            # generate fake output
            generated_data = self.generator(real_input, training=True)
            # reshape the data
            generated_data_reshape = tf.reshape(generated_data, [generated_data.shape[0], generated_data.shape[1], 1])
            d_fake_input = tf.concat([tf.cast(yc, tf.float32), generated_data_reshape], axis=1)
            real_y_reshape = tf.reshape(real_y, [real_y.shape[0], real_y.shape[1], 1])
            d_real_input = tf.concat([tf.cast(yc, tf.float32), tf.cast(real_y_reshape, tf.float32)], axis=1)

            fake_output = self.discriminator_c(d_fake_input, training=True)
            real_output = self.discriminator_c(d_real_input, training=True)

            disc_c_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_discriminator = disc_c_tape.gradient(disc_c_loss, self.discriminator_c.trainable_variables)
        self.discriminator_c_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator_c.trainable_variables))
        
        # Train the generator
        with tf.GradientTape() as gen_c_tape:
            # generate fake output
            generated_data = self.generator(real_input, training=True)
            # reshape the data
            generated_data_reshape = tf.reshape(generated_data, [generated_data.shape[0], generated_data.shape[1], 1])
            g_fake_input = tf.concat([tf.cast(yc, tf.float32), generated_data_reshape], axis=1)
            g_fake_output = self.discriminator_c(g_fake_input, training=True)
            gen_c_loss = self.generator_loss(g_fake_output)
            
        gradients_of_generator = gen_c_tape.gradient(gen_c_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        # Train the discriminator
        with tf.GradientTape() as disc_p_tape:
            # generate fake output
            generated_data_IDM = self.generator(real_input_IDM, training=True)
            # reshape the data
            generated_data_IDM_reshape = tf.reshape(generated_data_IDM, [generated_data_IDM.shape[0], generated_data_IDM.shape[1], 1])
            d_fake_input_IDM = tf.concat([tf.cast(yc_IDM, tf.float32), generated_data_IDM_reshape], axis=1)
            real_y_IDM_reshape = tf.reshape(real_y_IDM, [real_y_IDM.shape[0], real_y_IDM.shape[1], 1])
            d_real_input_IDM = tf.concat([tf.cast(yc_IDM, tf.float32), tf.cast(real_y_IDM_reshape, tf.float32)], axis=1)

            fake_output_IDM = self.discriminator_p(d_fake_input_IDM, training=True)
            real_output_IDM = self.discriminator_p(d_real_input_IDM, training=True)

            disc_p_loss = self.discriminator_loss(real_output_IDM, fake_output_IDM)

        gradients_of_discriminator_IDM = disc_p_tape.gradient(disc_p_loss, self.discriminator_p.trainable_variables)
        self.discriminator_p_optimizer.apply_gradients(zip(gradients_of_discriminator_IDM, self.discriminator_p.trainable_variables))
        
        # Train the generator
        with tf.GradientTape() as gen_p_tape:
            # generate fake output
            generated_data_IDM = self.generator(real_input_IDM, training=True)
            # reshape the data
            generated_data_IDM_reshape = tf.reshape(generated_data_IDM, [generated_data_IDM.shape[0], generated_data_IDM.shape[1], 1])
            g_fake_input_IDM = tf.concat([tf.cast(yc_IDM, tf.float32), generated_data_IDM_reshape], axis=1)
            g_fake_output_IDM = self.discriminator_p(g_fake_input_IDM, training=True)
            gen_p_loss = self.generator_loss(g_fake_output_IDM)
            
        gradients_of_generator_IDM = gen_p_tape.gradient(gen_p_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator_IDM, self.generator.trainable_variables))
        
        return disc_c_loss, gen_c_loss, disc_p_loss, gen_p_loss
        
    def train(self, X_train, y_train, yc_train, X_train_IDM, y_train_IDM, yc_train_IDM, opt):
        data = X_train, y_train, yc_train
        data_IDM = X_train_IDM, y_train_IDM, yc_train_IDM
        idx_range = np.arange(0, X_train.shape[0]-self.batch_size, 1)
        idx_0 = 0
        losses = []
        epochs = opt["epoch"]

        for epoch in range(epochs):
            #start = time.time()
            if idx_0 >= idx_range[-1]+1:
                idx_0 = 0
            d_c_loss, g_c_loss, d_p_loss, g_p_loss = self.train_step(data, data_IDM, idx_0)
            idx_0 += 128

            #print('Epoch %d/%d' %(epoch+1, epochs))
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            #if (epoch + 1) % int(X_train.shape[0]/self.batch_size) == 0:
            #    tf.keras.models.save_model(generator, 'genp/gen_%d.h5' % (epoch+1))
            #    print('epoch', epoch+1, 'd_c_loss', d_c_loss.numpy(), 'g_c_loss', g_c_loss.numpy(), 'd_p_loss', d_p_loss.numpy(), 'g_p_loss', g_p_loss.numpy())
                
            #    # Output a sample of generated image
            #    self.sample_images(generator)

            # Save the model every 100 epochs
            if (epoch + 1) % 100 == 0:
                tf.keras.models.save_model(generator, 'gen/gen_%d.h5' % (epoch+1))
                #print('epoch', epoch+1, 'd_c_loss', d_c_loss.numpy(), 'g_c_loss', g_c_loss.numpy(), 'd_p_loss', d_p_loss.numpy(), 'g_p_loss', g_p_loss.numpy())
                # Output a sample of generated image
                self.sample_images(generator, epoch)

if __name__ == '__main__':
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    feature_size_in = X_train.shape[2]
    feature_size_out = yc_train.shape[2]
    img_shape = (output_dim, feature_size_out)
    xy_shape = (input_dim+output_dim, feature_size_out)

    ## For Bayesian
    opt = {"lr": 0.00005, "epoch": 10000, 'bs': 128}
    
    generator = Generator(input_dim, output_dim, feature_size_in, feature_size_out, img_shape)
    discriminator_c = Discriminator_c(xy_shape)
    discriminator_p = Discriminator_p(xy_shape)
    gan = GAN(generator, discriminator_c, discriminator_p, opt)
    gan.train(X_train, y_train, yc_train, X_train_IDM, y_train_IDM, yc_train_IDM, opt)