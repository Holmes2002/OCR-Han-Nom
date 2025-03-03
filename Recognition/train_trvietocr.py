from trvietocr.mix_trocr_vietocr import TrVietOCR
from trvietocr.load_config_trvietocr import Cfg
from transformers import TrOCRProcessor
from trvietocr.dataset import ICOCRDataset, TrVietOCRDataset, Collator
from torch.utils.data import DataLoader
from tqdm import tqdm
from trvietocr.utils import get_lr, cosine_lr
from transformers import AdamW
import os
import torch
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import GradScaler
from torch import autocast

HanNom_vocab = open('/home1/vudinh/NomNaOCR/HanNom_Vocab/HanNom_vocab.txt').read().splitlines()
config = Cfg.load_config_from_file("./config/trvietocr.yml")
os.makedirs(config.ckpt_save_path, exist_ok = True)
log_file = open(f'{config.ckpt_save_path}/log_train.txt', 'w')

model = TrVietOCR(vocab_size=len(HanNom_vocab), 
                  width_size=config.width_size, 
                  height_size=config.height_size, 
                  max_length_token=config.max_length_token,
                  vietocr_pretrained=config.vietocr_pretrained,
                  decoder_vietocr_pretrained=config.decoder_vietocr_pretrained,
                  encoder_trocr_pretrained=config.encoder_trocr_pretrained,
                  fc_vietocr_pretrained=config.fc_vietocr_pretrained,
                  vocab = HanNom_vocab
)
if config.ckpt != '':
    model.load_state_dict(torch.load(config.ckpt))
processor_pretrained_path = config.processor_pretrained_path
processor                 = TrOCRProcessor.from_pretrained(processor_pretrained_path)
# model_parameters = filter(lambda p: p.requires_grad, model.decoder.parameters())
# params_count = sum([p.numel() for p in model_parameters])
# print(f"Total number of trainable parameters: {params_count}")


check_point = torch.load('/home1/vudinh/NomNaOCR/icocr/union/ckpt_384_v3/epoch_2.pth')
encoder_state = {k.replace('decoder.',''):v for k,v in check_point.items() if 'decoder' in k}
model.decoder.load_state_dict(encoder_state, strict = True)
encoder_state = {k.replace('fc.',''):v for k,v in check_point.items() if 'fc' in k}
model.fc.load_state_dict(encoder_state, strict = False)

# print(model)
# assert False
file_train = '/home1/vudinh/NomNaOCR/PaddleOCR-release-2.6/dataset/detection/Patches/Train.txt'
train_dataset                = TrVietOCRDataset(data_dir= file_train,
                                            processor= processor,
                                            data_type= "train",
                                            max_target_length = config.max_length_token,
                                            load_file= True,
                                            chars=HanNom_vocab)
masked_language_model = True
collate_fn = Collator(masked_language_model)
train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=config.shuffle, num_workers= 32, collate_fn=collate_fn)
num_batches      = len(train_dataloader)

## hype params
lr = 5e-5
model.encoder.requires_grad_(False)
model.decoder.requires_grad_(False)
model.fc.requires_grad_(False)

# model.load_state_dict(torch.load("/home1/vudinh/NomNaOCR/icocr/trvietocr/ckpt/epoch_2.pth"), strict = True)
params      = [p for name, p in model.named_parameters() if p.requires_grad]
params_name = [name for name, p in model.named_parameters() if p.requires_grad]
print(model)
print(params_name)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params_count = sum([p.numel() for p in model_parameters])
print(f"Total number of trainable parameters: {params_count}")
log_file.write(str(params_name) + '\n')
log_file.write(f"Total number of trainable parameters: {params_count}"+ '\n')
optimizer = AdamW(params, lr=lr)

scheduler = cosine_lr(optimizer, lr, config.warmup_length, config.num_epochs * num_batches)
device = config.device
model.to(device)
loss_fct = CrossEntropyLoss()
scaler = GradScaler()

best_acc = 0
best_cer = 1e3
best_epoch = 0
FP_16 = True
for epoch in range(config.num_epochs):
    print("[INFO] Epoch: {}".format(epoch))
    model.train()
    train_loss = 0.0
    epoch += config.resume_epoch
    for i, batch in enumerate(tqdm(train_dataloader)):
        step = i + epoch * num_batches
        lr = scheduler(step)
        # print(batch.items())
        for k,v in batch.items():
            batch[k] = v.to(device)
        if not FP_16:
            logits = model(**batch)
            logits = logits.view(-1, logits.size(2))
            loss = loss_fct(logits, batch["tgt_output"].view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        else:
            with autocast( dtype=torch.float16, device_type = 'cuda'):
                logits = model(**batch)
                logits = logits.view(-1, logits.size(2))
                loss = loss_fct(logits, batch["tgt_output"].view(-1)) 
            # Scales the loss, and backward pass
            scaler.scale(loss).backward()
            
            # Unscales gradients and performs optimizer step
            scaler.step(optimizer)
            scaler.update()

        train_loss += loss.item()
        if (i+1) % 100 == 0:
           iter_loss = train_loss/(i+1)
           log_file.write(f"[INFOR] Loss in iter {i+1}: {iter_loss} at epoch {epoch} with LR {get_lr(optimizer)}\n")
           print(f"[INFOR] Loss in iter {i+1}: {iter_loss} at epoch {epoch} with LR {get_lr(optimizer)}")
        if (i+1) % 20000 == 0:
            os.makedirs(config.ckpt_save_path, exist_ok = True)
            dir_ckpt = os.path.join(config.ckpt_save_path, f"epoch_{epoch+1}_{i}.pth")
            torch.save(model.state_dict(), dir_ckpt)
    #         continue
    epoch_loss =  train_loss/len(train_dataloader)
    print(f"[INFO] Loss after epoch {epoch}:", epoch_loss)
    os.makedirs(config.ckpt_save_path, exist_ok = True)
    dir_ckpt = os.path.join(config.ckpt_save_path, "epoch_{}.pth".format(epoch+1))
    torch.save(model.state_dict(), dir_ckpt)
# for epoch in range(config.num_epochs):
#     print("[INFO] Epoch: {}".format(epoch))
#     model.train()
#     train_loss = 0.0
#     epoch += config.resume_epoch
#     for i, batch in enumerate(tqdm(train_dataloader)):
#         step = i + epoch * num_batches
#         lr = scheduler(step)
#         # print(batch.items())
#         for k,v in batch.items():
#             batch[k] = v.to(device)
        
#         logits = model(**batch)
#         logits = logits.view(-1, logits.size(2))
#         loss = loss_fct(logits, batch["tgt_output"].view(-1))
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#         train_loss += loss.item()
#         if (i+1)%50==0:
#             iter_loss = train_loss/(i+1)
#             print("[INFO] Loss in iter {}: {}".format(i+1, iter_loss))
#             print("[INFO] Learning rate  : {}".format(get_lr(optimizer)))
#         if (i+1)%4000==0:
#             # dir_ckpt = "./weights/epoch/epoch_{}_iter_{}".format(epoch+1, i+1)
#             dir_ckpt = os.path.join(config.ckpt_save_path, "epoch_{}_iter_{}.pth".format(epoch+1, i+1))
#             # os.mkdir(dir_ckpt)
#             # model.save_pretrained(dir_ckpt)
#             torch.save(model.state_dict(), dir_ckpt)

#     epoch_loss =  train_loss/len(train_dataloader)
#     print(f"[INFO] Loss after epoch {epoch}:", epoch_loss)
#     dir_ckpt = os.path.join(config.ckpt_save_path, "epoch_{}.pth".format(epoch+1))
#     torch.save(model.state_dict(), dir_ckpt)



# # loss = None
# # if labels is not None:
# #     logits = decoder_outputs.logits if return_dict else decoder_outputs
# #     logits = logits.view(-1, logits.size(2))
# #     loss_fct = CrossEntropyLoss()
# #     logits = logits.reshape(-1, self.fc.out_features)
# #     loss = loss_fct(logits, labels.view(-1))
# # return loss   