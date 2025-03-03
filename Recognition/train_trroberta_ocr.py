from union.mix_union_vietocr import UnionVietOCR, UnionRoBerta, UnionRoBerta, TrRoBerta
from union.dataset import UniVietOCRDataset, Collator, UnionROBERTtaDataset, Collator_Roberta
from torch.utils.data import DataLoader
from trvietocr.load_config_trvietocr import Cfg
from transformers import AdamW
import os
import torch
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import GradScaler
from torch import autocast
from trvietocr.utils import get_lr, cosine_lr
from tqdm import tqdm
from icocr_infer import eval
from torch.cuda.amp import autocast, GradScaler
from icocr_roberta_infer import test_train
scaler = GradScaler(growth_interval=100)
from transformers import AutoTokenizer, AutoModel

config = Cfg.load_config_from_file("./config/trrobertaocr.yml")
vocab = config.vocab
device = config.device
HanNom_vocab = open('/home1/vudinh/NomNaOCR/HanNom_Vocab/unique_vocab.txt').read().splitlines()
# HanNom_vocab = open('vocab_tmp.txt').read().splitlines()
tokenizer = AutoTokenizer.from_pretrained("KoichiYasuoka/roberta-small-japanese-aozora-char")
tokenizer = tokenizer.train_new_from_iterator(HanNom_vocab, vocab_size=len(HanNom_vocab))
os.makedirs(config.ckpt_save_path, exist_ok = True)
log_file = open(f'{config.ckpt_save_path}/log_train.txt', 'w')
model = TrRoBerta(max_length_token=config.max_length_token,
                     img_width=config.width_size,
                     img_height=config.height_size,
                     patch_size=config.patch_size,
                     tokenizer = tokenizer,
                     type = 'v2',
                     embed_dim_vit=config.embed_dim_vit
                     )
if config.ckpt != '':
    model.load_state_dict(torch.load(config.ckpt))
# tokenizer = AutoTokenizer.from_pretrained("KoichiYasuoka/roberta-small-japanese-aozora-char")
checkpoint = torch.load("/home1/vudinh/NomNaOCR/icocr/union_roberta/ckpt_320x48/epoch_1_72999.pth", map_location=torch.device("cuda"))
# print(model)
# encoder_state = {k.replace('decoder.',''):v for k,v in checkpoint.items() if 'decoder' in k}
# model.decoder.load_state_dict(encoder_state, strict = False)
# encoder_state = {k.replace('enc_to_dec_proj.',''):v for k,v in checkpoint.items() if 'enc_to_dec_proj' in k}
# model.enc_to_dec_proj.load_state_dict(encoder_state, strict = True)

model.load_state_dict(torch.load("/home1/vudinh/NomNaOCR/icocr/union_roberta/ckpt_320x48/epoch_1_72999.pth"), strict = False)
model.to(device)
masked_language_model = True
# collate_fn = Collator(masked_language_model)
# train_data_dir = '/home1/vudinh/NomNaOCR/Text-Generator-main/Train.txt'
train_data_dir = '/home1/vudinh/NomNaOCR/PaddleOCR-release-2.6/dataset/detection/Patches/Train_with_OpenSource.txt'
val_data_dir = '/home1/vudinh/NomNaOCR/PaddleOCR-release-2.6/dataset/detection/Patches/Val_real.txt'


# val_dataset           = UniVietOCRDataset(data_dir= val_data_dir,
#                                             data_type= "val",
#                                             max_target_length = config.max_length_token,
#                                             img_size=(config.width_size, config.height_size),
#                                             load_file= True,
#                                             chars=HanNom_vocab)
# val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=config.shuffle, num_workers=config.num_workers, collate_fn=collate_fn)
train_dataset           = UnionROBERTtaDataset(data_dir= train_data_dir,
                                            data_type= "train",
                                            max_target_length = config.max_length_token,
                                            img_size=(config.width_size, config.height_size),
                                            load_file= True,
                                            chars=HanNom_vocab, 
                                            is_finetune_model = True)
collate_fn = Collator_Roberta(HanNom_vocab)
train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=8, collate_fn = collate_fn)
# eval(train_dataloader, model, config, HanNom_vocab)
num_batches      = len(train_dataloader)

## hype params
lr = 2e-5
# model.vit_model.requires_grad_(False)
model.decoder.requires_grad_(False)
# model.decoder.output_projection.requires_grad_(True)
# model.fc.requires_grad_(False)
# model.enc_to_dec_proj.requires_grad_(False)
# model.embed_tgt.requires_grad_(False)

# model.vit_model.pos_embed.requires_grad_(True)

params      = [p for name, p in model.named_parameters() if p.requires_grad] 
params_name = [name for name, p in model.named_parameters() if p.requires_grad]

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params_count = sum([p.numel() for p in model_parameters])
log_file.write(str(params_name) + '\n')
log_file.write(f'Number of Parameters: {params_count}' + '\n')
optimizer = AdamW(params, lr=lr)
scheduler = cosine_lr(optimizer, lr, config.warmup_length, config.num_epochs * num_batches)
device = config.device
model.to(device)
loss_fct = CrossEntropyLoss()
scaler = GradScaler()
print(model)
print(params_name)
print(f'Number of Parameters: {params_count}' + '\n')
best_acc = 0
best_cer = 1e3
best_epoch = 0
step_loss = 100
file_1 = open('file_2.txt', 'w')
for epoch in range(config.num_epochs):
    print("[INFO] Epoch: {}".format(epoch))
    model.train()
    train_loss = 0.0
    epoch += config.resume_epoch
    # avg_acc, avg_cer = eval(val_dataloader, model, config, HanNom_vocab)
    # print(avg_acc, avg_cer)
    # assert False
    for iter, batch in enumerate(tqdm(train_dataloader)):
        step = iter + epoch * num_batches
        lr = scheduler(step)
        if not config.use_fp16:
            for k,v in batch.items():
                batch[k] = v.to(device)
            # decoded_texts, labels, acc = model.inference(batch['img'], batch['tgt_output'])
            # model.inference(batch['img'], batch['tgt_output'])
            # assert False
            if False:
                logits = model(**batch)
                loss = loss_fct(logits.view(-1, logits.size(-1)), batch["tgt_output"].view(-1))
            else:
                loss, logits = model(**batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        else:
            with autocast( dtype=torch.float16):
                logits = model(**batch)
                logits = logits.view(-1, logits.size(2))
                loss = loss_fct(logits, batch["tgt_output"].view(-1)) 
            # Scales the loss, and backward pass
            scaler.scale(loss).backward()
            
            # Unscales gradients and performs optimizer step
            scaler.step(optimizer)
            scaler.update()

        train_loss += loss.item()
        if (iter+1) % step_loss == 0:
           iter_loss = train_loss/step_loss
           decoded_texts, labels, acc = model.inference(batch['img'], batch['tgt_output'])
           if acc > best_acc: best_acc = acc
           log_file.write(f"[INFOR] Loss in iter {iter+1}: {iter_loss} at epoch {epoch} with LR {get_lr(optimizer)} with best_acc {best_acc} \n")
           log_file.write(f'Predict {epoch} {iter}: {decoded_texts}\n')
           log_file.write(f'Lable {epoch} {iter}: {labels}\n')
        #    img_sample = batch['img'][0].unsqueeze(0)
        #    text = model.generate(img_sample)
        #    labels = tokenizer.decode(list(batch['input_ids'][0]))
           print(f"[INFOR] Loss in iter {iter+1}: {iter_loss} at epoch {epoch} with LR {get_lr(optimizer)} ")
           train_loss = 0
        #    file_1.write(''.join(['tgt_output ', str(tgt_output), ' iter :',str(step), '\n']))
        #    file_1.write(''.join(['logits ', str(logits), ' iter :',str(step), '\n']))

        if (iter+1) % 1000 == 0:
            
            dir_ckpt = os.path.join(config.ckpt_save_path, f"epoch_{epoch+1}_{iter}.pth")
            torch.save(model.state_dict(), dir_ckpt)
    # if True:
    #     try:
    #         avg_acc, avg_cer, avg_score = eval(val_dataloader, model, config, HanNom_vocab)
    #         if best_acc < avg_acc and best_cer > avg_cer:
    #             best_acc, best_cer = max(best_acc, avg_acc), min(best_cer, avg_cer)
    #             best_epoch = epoch
    #         print(f"[INFO] Loss in iter {i+1}: {iter_loss} at epoch {epoch} with LR {get_lr(optimizer)} | Best Acc {best_acc} | Best_CER {best_cer}")
    #         log_file.write(f"[INFO] Loss in iter {i+1}: {iter_loss} at epoch {epoch} with LR {get_lr(optimizer)} | Best Acc {best_acc} | Best_CER {best_cer} at {best_epoch} | Avg Score {avg_score}\n ")
    #     except:
    #         pass
    # epoch_loss =  train_loss/len(train_dataloader)
    # print(f"[INFO] Loss after epoch {epoch}:", epoch_loss)
    os.makedirs(config.ckpt_save_path, exist_ok = True)
    dir_ckpt = os.path.join(config.ckpt_save_path, "epoch_{}.pth".format(epoch+1))
    torch.save(model.state_dict(), dir_ckpt)