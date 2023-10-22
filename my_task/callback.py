from keras.callbacks import Callback, ModelCheckpoint
import numpy as np
import pandas as pd
from keras import backend as K
import os
from keras.callbacks import Callback
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
import smiles_process


class WeightAnnealer_epoch(Callback):        #调整变分自动编码器（VAE）的权重
    '''Weight of variational autoencoder scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            weight for the VAE (float).
        Currently just adjust kl weight, will keep xent weight constant
    '''

    def __init__(self, schedule, weight, weight_orig, weight_name):
        super(WeightAnnealer_epoch, self).__init__()
        self.schedule = schedule
        self.weight_var = weight
        self.weight_orig = weight_orig
        self.weight_name = weight_name

    def on_epoch_begin(self, epoch, logs=None):
        if logs is None:
            logs = {}
        new_weight = self.schedule(epoch)
        new_value = new_weight * self.weight_orig
        print("Current {} annealer weight is {}".format(self.weight_name, new_value))
        assert type(
            new_weight) == float, 'The output of the "schedule" function should be float.'
        K.set_value(self.weight_var, new_value)


# Schedules for VAEWeightAnnealer
def no_schedule(epoch_num):
    return float(1)


def sigmoid_schedule(time_step, slope=1., start=None):
    return float(1 / (1. + np.exp(slope * (start - float(time_step)))))


def sample(a, temperature=0.01):
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


class EncoderDecoderCheckpoint(ModelCheckpoint):      #保存编码器和解码器的模型

    def __init__(self, encoder_model, decoder_model, params, prop_to_monitor='val_x_pred_categorical_accuracy', save_best_only=True, monitor_op=np.greater, monitor_best_init=-np.Inf):

        super(ModelCheckpoint, self).__init__()
        self.save_best_only = save_best_only
        self.monitor = prop_to_monitor
        self.monitor_op = monitor_op
        self.best = monitor_best_init
        self.verbose = 1
        self.encoder = encoder_model
        self.decoder = decoder_model

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.save_best_only:
            current = logs.get(self.monitor)
            if self.monitor_op(current, self.best):
                if self.verbose > 0:
                    print('Epoch %05d: %s improved from %0.5f to %0.5f, saving model' % (epoch, self.monitor, self.best, current))
                self.best = current
                self.encoder.save(os.path.join(self.p['checkpoint_path'], 'encoder_{}.h5'.format(epoch)))
                self.decoder.save(os.path.join(self.p['checkpoint_path'], 'decoder_{}.h5'.format(epoch)))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: %s did not improve' % (epoch, self.monitor))
        else:
            if self.verbose > 0:
                print('Epoch %05d: saving model to ' % (epoch))
            self.encoder.save(os.path.join(self.p['checkpoint_path'], 'encoder_{}.h5'.format(epoch)))
            self.decoder.save(os.path.join(self.p['checkpoint_path'], 'decoder_{}.h5'.format(epoch)))
            
#读取SMILES：首先，我们需要读取您提供的包含10个相似SMILES的txt文件。
#编码SMILES：使用VAE的编码器将这10个SMILES编码成隐空间中的点。
#计算距离：为了衡量VAE对细微差异的敏感性，我们可以在隐空间中计算每两个SMILES对应的编码之间的欧式距离。理论上，如果两个SMILES非常相似，但在某些关键特征上有所不同，那么它们在隐空间中的距离应该反映这种差异。
#可视化：我们可以使用热图来可视化这10个SMILES在隐空间中的距离矩阵。随着训练的进行，我们应该能看到距离矩阵的变化，这可以帮助我们了解模型对SMILES的敏感性是如何发展的。            
class SmilesSensitivityCallback(Callback):     #监测模型对SMILES的敏感性
    def __init__(self, smiles_file, encoder, max_len, char_indices, nchars, padding='right'):
        super(SmilesSensitivityCallback, self).__init__()
        
        with open(smiles_file, 'r') as f:
            smiles_list = [line.strip() for line in f.readlines()]
        
        # Filter and verify the smiles
        self.smiles_list = [smile for smile in smiles_list if smiles_processing.verify_smiles(smile)]
        self.encoder = encoder
        self.max_len = max_len
        self.char_indices = char_indices
        self.nchars = nchars
        self.padding = padding

    def on_epoch_end(self, epoch, logs=None):
        # Convert SMILES to one-hot encodings
        one_hot_smiles = smiles_processing.smiles_to_hot(self.smiles_list, self.max_len, self.padding, self.char_indices, self.nchars)
        
        # Use the encoder to encode the SMILES
        encoded_smiles = self.encoder.predict(one_hot_smiles)
        
        # Calculate the distance matrix
        num_smiles = len(self.smiles_list)
        distance_matrix = np.zeros((num_smiles, num_smiles))
        for i in range(num_smiles):
            for j in range(num_smiles):
                distance_matrix[i, j] = np.linalg.norm(encoded_smiles[i] - encoded_smiles[j])
        
        # Visualize the distance matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(distance_matrix, annot=True, fmt=".2f")
        plt.title(f"Epoch: {epoch+1}")
        plt.savefig(f"distance_matrix_epoch_{epoch+1}.png")
        plt.close()
