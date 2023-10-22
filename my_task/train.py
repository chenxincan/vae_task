#from callback import SmilesSensitivityCallback
# 其他必要的引入，如 keras.models, keras.layers 等

# 定义或加载模型、编码器等

# 初始化回调
#smiles_sensitivity_cb = SmilesSensitivityCallback('path_to_your_smiles_file.txt', encoder, max_len, char_indices, nchars)

# 开始训练
#model.fit(your_data, your_labels, epochs=your_epochs, batch_size=your_batch_size, callbacks=[..., smiles_sensitivity_cb])




import os
import numpy as np
from keras import backend as K
from keras.optimizers import Adam
from model import MoleculeVAE
from data_preprocessing import DataPreprocessing
from callback import WeightAnnealer_epoch, EncoderDecoderCheckpoint, SmilesSensitivityCallback
from parameters import load_params

# 加载参数
params = load_params(param_file='parameters.py')

with open('params.json') as f:
    json_params = json.load(f)
    params.update(json_params)

# 加载数据
data_processor = DataPreprocessing(smiles_file='smiles.txt', params=params)
X_train, X_val = data_processor.preprocess_data()

# 初始化模型
molecule_vae = MoleculeVAE(params)

# 编译模型
optimizer = Adam(lr=params['lr'], beta_1=0.9)
molecule_vae.model.compile(optimizer=optimizer, loss=params['loss'])

# 定义回调函数
weight_annealer = WeightAnnealer_epoch(schedule=lambda epoch: np.exp(-epoch/10), weight=K.variable(1.0), weight_orig=1.0, weight_name='KL Weight')
encoder_decoder_checkpoint = EncoderDecoderCheckpoint(encoder_model=molecule_vae.encoder, decoder_model=molecule_vae.decoder, params=params)
smiles_sensitivity_callback = SmilesSensitivityCallback(smiles_file='smiles.txt', encoder=molecule_vae.encoder, max_len=params['max_len'], char_indices=data_processor.char_indices, nchars=len(data_processor.chars))

callbacks = [weight_annealer, encoder_decoder_checkpoint, smiles_sensitivity_callback]

# 训练模型
molecule_vae.model.fit(X_train, X_train,
                       epochs=params['epochs'],
                       batch_size=params['batch_size'],
                       validation_split=params['val_split'],
                       callbacks=callbacks)

# 保存模型
molecule_vae.model.save('/Users/chenxincan/Desktop/DL/my/my_task/output/molecule_vae_model.h5')

# 在这里，你还可以添加代码来评估模型和在测试集上进行预测

