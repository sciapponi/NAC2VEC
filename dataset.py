import torchaudio 
import torch 
import ast 
import torch.nn.functional as F
from torch.utils.data import Dataset

class FluentSpeechCommands(Dataset):
    def __init__(self, data_path, max_len_audio=64000, train="train", batch_size=16):
        if not isinstance(train, bool) and train not in ("train", "valid", "test"):
            raise ValueError(f"`train` arg ({train}) must be a bool or train/valid/test.")
        
        self.train = train
        self.dataset = torchaudio.datasets.FluentSpeechCommands(data_path, train)
        self.max_len_audio = max_len_audio
        self.batch_size = batch_size
        
        # Pad dataset to make it a multiple of the batch size
        self.original_len = len(self.dataset)
        # self.original_len = 400
        self.padded_len = (
            (self.original_len + self.batch_size - 1) // self.batch_size
        ) * self.batch_size
    
    def pad_or_trim(self, waveform):
        current_samples = waveform.shape[1]
        if current_samples > self.max_len_audio:
            # Trim: take the first target_samples
            return waveform[..., :self.max_len_audio]
        elif current_samples < self.max_len_audio:
            # Pad: add zeros to the end
            padding = self.max_len_audio - current_samples
            return F.pad(waveform, (0, padding))
        else:
            return waveform

    def _one_hot_labels(self, int_label):
        return F.one_hot(torch.tensor(int_label), num_classes=len(self.class_ids))
    
    def __len__(self):
        return self.padded_len
    
    def __getitem__(self, index):
        if index >= self.original_len:
            # Return dummy data for padded indices
            dummy_wav = torch.zeros(1, self.max_len_audio)  # Adjust shape as needed
            dummy_label = torch.zeros(len(self.class_ids))  # Dummy one-hot label
            return dummy_wav, dummy_label
        
        # Return actual data
        item = self.dataset[index]
        wav = self.pad_or_trim(item[0])
        action, obj, location = item[-3:]
        label_key = action + obj + location
        label = self._one_hot_labels(self.class_ids[label_key])
        return wav, label

    @property
    def class_ids(self):
        return {
            'change languagenonenone': 0,
            'activatemusicnone': 1,
            'activatelightsnone': 2,
            'deactivatelightsnone': 3,
            'increasevolumenone': 4,
            'decreasevolumenone': 5,
            'increaseheatnone': 6,
            'decreaseheatnone': 7,
            'deactivatemusicnone': 8,
            'activatelampnone': 9,
            'deactivatelampnone': 10,
            'activatelightskitchen': 11,
            'activatelightsbedroom': 12,
            'activatelightswashroom': 13,
            'deactivatelightskitchen': 14,
            'deactivatelightsbedroom': 15,
            'deactivatelightswashroom': 16,
            'increaseheatkitchen': 17,
            'increaseheatbedroom': 18,
            'increaseheatwashroom': 19,
            'decreaseheatkitchen': 20,
            'decreaseheatbedroom': 21,
            'decreaseheatwashroom': 22,
            'bringnewspapernone': 23,
            'bringjuicenone': 24,
            'bringsocksnone': 25,
            'change languageChinesenone': 26,
            'change languageKoreannone': 27,
            'change languageEnglishnone': 28,
            'change languageGermannone': 29,
            'bringshoesnone': 30
        }




if __name__=="__main__":
    dataset = FluentSpeechCommands(data_path="/home/ste/Datasets")
    print(dataset[0][0].shape)
    # max = 0
    # for i in dataset:
    #     if max < int(i[0].shape[-1]):
    #         max = int(i[0].shape[-1])
    #         # print(max)

        
    # print(max)