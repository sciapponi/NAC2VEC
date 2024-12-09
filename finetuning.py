import lightning as L
import torch 
from torch import nn 
from feature_extraction import NACFeatureExtractionChunked
from transformers import Wav2Vec2Model, AutoConfig
import numpy as np 
from torchmetrics import Accuracy, F1Score, Recall
from dataset import FluentSpeechCommands
from itertools import chain
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import Trainer

class NAC2Vec(L.LightningModule):

    def __init__(self, nac_path, num_classes=31):
        super().__init__()
        self.fe = NACFeatureExtractionChunked(nac_path)
        self._device = "cuda"
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", 
                                              torch_dtype=torch.float32, 
                                              ).to(self._device)
        self.projection = nn.Linear(256,model.config.hidden_size).to(self._device)
        self.w2vencoder = model.encoder 
        self.w2vencoder.train()
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.AdaptiveAvgPool1d(1),  # Pool temporal dimension
            nn.Flatten(),
            nn.Linear(model.config.hidden_size, num_classes),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            # nn.Linear(256, num_classes)
        ).to(self._device)
        del model 

        self.loss = nn.CrossEntropyLoss()
        self.validation_step_outputs = self.reset_valid_outputs()

        self.accuracy = Accuracy("multiclass", num_classes=num_classes)
        self.recall = Recall("multiclass", num_classes=num_classes)
        self.f1 = F1Score("multiclass", num_classes=num_classes)
        self.validation_step_outputs = self.reset_valid_outputs()

    def reset_valid_outputs(self):
        return {"accuracy":[],
                "recall":[],
                "f1":[]}
         

    def configure_optimizers(self):
        # Create optimizer
        optimizer = torch.optim.AdamW(
            chain(self.projection.parameters(),
                  self.w2vencoder.parameters(),
                  self.classifier.parameters())
            , 
            lr=3e-4,  # default learning rate
            weight_decay=0.01  # optional weight decay
        )
        
        # Create learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=10,  # total number of training epochs
            eta_min=1e-7  # minimum learning rate
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # called once per epoch
                'frequency': 1
            }
        }
    
    def forward(self, x):
        x = self.fe.extract(x.cpu().numpy()).float().to(self._device)
        # print(x.shape)
        x = self.projection(x)
        x = self.w2vencoder(x)
        # print(x.last_hidden_state.shape)
        x = x.last_hidden_state.transpose(1,2)
        x = self.classifier(x)
        # print(x.shape)
        return x

    def training_step(self, batch, batch_idx):
        wavs, labels = batch 
        outputs = self(wavs)
        loss = self.loss(outputs, labels.float())
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        wavs, labels = batch 
        outputs = self(wavs)
        # print(torch.argmax(outputs, dim=-1))
        # print(torch.argmax(labels, dim=-1))
        # print(self.accuracy(torch.argmax(outputs, dim=-1), torch.argmax(labels, dim=-1)))
        self.validation_step_outputs["accuracy"].append(self.accuracy(torch.argmax(outputs, dim=-1), torch.argmax(labels, dim=-1)))
        self.validation_step_outputs["recall"].append(self.recall(torch.argmax(outputs, dim=-1), torch.argmax(labels, dim=-1)))
        self.validation_step_outputs["f1"].append(self.f1(torch.argmax(outputs, dim=-1), torch.argmax(labels, dim=-1)))
    
    def on_validation_epoch_end(self):
        self.log("val/accuracy", torch.tensor(self.validation_step_outputs["accuracy"]).mean())
        self.log("val/recall", torch.tensor(self.validation_step_outputs["recall"]).mean())
        self.log("val/f1", torch.tensor(self.validation_step_outputs["f1"]).mean())
        self.validation_step_outputs = self.reset_valid_outputs()
    
    def train_dataloader(self):
        return self._make_dataloader(train=True)
    
    def val_dataloader(self):
        return self._make_dataloader(train=False)
    
    def _make_dataloader(self, train):

        def collate(examples):
            # print(len(examples))
            # exit()
            # stacked = torch.stack(examples)
            wavs, labels = zip(*examples)
    
            # Convert to tensors
            wavs = torch.stack(list(wavs))
            labels = torch.stack(labels)
            
            return wavs, labels
        
        
        dataset = FluentSpeechCommands("/home/ste/Datasets", 64000, "train" if train else "test")
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=train,
                                                collate_fn=collate, num_workers=19, pin_memory=True, persistent_workers=True)

        return loader 
    
if __name__=="__main__":
    n = NAC2Vec("b_16encoder_rvq_fp16.onnx").to("cuda")
    # print(n(np.ones((16,1,120000))))
    logger = CSVLogger("logs", name="adamw3e-5")
    trainer = Trainer(logger = logger,
                      devices=1,
                      accelerator="gpu",
                      max_epochs=10)
    trainer.fit(n)