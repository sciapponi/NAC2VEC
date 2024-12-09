import onnxruntime
import torch 
from torch import nn 
import numpy as np

class NACFeatureExtraction(nn.Module):

    def __init__(self, path):

        self.session = onnxruntime.InferenceSession(path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    def extract(self, input):

        input = {"input":input.astype(np.float16)}

        return torch.from_numpy(self.session.run(None, input)[0])
    
class NACFeatureExtractionChunked(nn.Module):
    def __init__(self, path):
        super().__init__() 
        self.session = onnxruntime.InferenceSession(path)

    def extract(self, input):
        # If input is smaller than or equal to 16000 samples, process directly
        if input.shape[-1] <= 16000:
            input_dict = {"input": input.astype(np.float16)}
            return torch.from_numpy(self.session.run(None, input_dict)[0])
        
        # Otherwise, split into chunks and stack results
        chunks = []
        for i in range(0, input.shape[-1], 16000):
            # print(i)
            # Take a chunk of up to 16000 samples
            chunk = input[...,i:i+16000]
            if chunk.shape[-1]<16000:

                chunk = np.pad(chunk, 
                      pad_width=((0,0), (0,0), (0, 16000 - chunk.shape[-1])), 
                      mode='constant', 
                      constant_values=0)
            # Prepare chunk for inference
            chunk_dict = {"input": chunk.astype(np.float16)}
            
            # Run inference on chunk and store result
            chunk_result = self.session.run(None, chunk_dict)[0]
            chunks.append(torch.from_numpy(chunk_result))
        
        # Stack all chunks along the first dimension
        return torch.cat(chunks, dim=1)
    
if __name__=="__main__":
    fe = NACFeatureExtractionChunked("encoder_rvq_fp16.onnx")
    print(fe.extract(np.ones((1,1,270000))).shape)