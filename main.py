import torch 
from transformers import Wav2Vec2Model, AutoConfig
import onnxruntime
import numpy as np 
from feature_extraction import NACFeatureExtractionChunked
device = "cuda"
# config = AutoConfig.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
# print(config)
# exit()
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self", torch_dtype=torch.float16, attn_implementation="flash_attention_2").to(device)

encoder = model.encoder 
print(model.config.hidden_size)
exit()
del model 
print(encoder)

model = encoder
param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

print(f"Number of parameters: {(param_size + buffer_size)}")
size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.3f}MB'.format(size_all_mb))

fe = NACFeatureExtractionChunked("/home/ste/Code/NAC2vec/encoder_rvq_fp16.onnx")
print(fe.extract(np.zeros((1,1,120000))).shape)

