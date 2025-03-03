import torch.nn as nn 
from union.vit_model import VisionTransformer
import torch
from vietocr.tool.config import Cfg
from vietocr.tool.translate import build_model
import math
from transformers import AutoTokenizer, AutoModel, AutoConfig
from union.decoder import init_decoder
from transformers import AutoTokenizer,AutoModelForMaskedLM, VisionEncoderDecoderModel, AutoModelForCausalLM
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from PIL import Image
import cv2
import collections.abc


def hugging_face_decoder_init(huggingface_repo, tokenizer):
    decoder_config = AutoConfig.from_pretrained(huggingface_repo)
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True

    # Build the decoder model
    decoder = AutoModelForCausalLM.from_pretrained(
        huggingface_repo, 
        config=decoder_config
    )
    decoder.roberta.embeddings.word_embeddings = nn.Embedding(len(tokenizer), decoder.config.hidden_size, padding_idx=0)
    # Disable position embeddings (set to zero)
    decoder.roberta.embeddings.position_embeddings = nn.Embedding(130, decoder.config.hidden_size, padding_idx=0)  # Assuming PE size is unchanged
    # Update lm_head to match new vocabulary size
    decoder.lm_head.decoder = nn.Linear(decoder.config.hidden_size, len(tokenizer), bias=True)
    # Update the configuration to reflect the changes
    decoder.config.vocab_size = len(tokenizer)
    decoder.config.bos_token_id = tokenizer.cls_token_id  # Assuming cls_token_id is the start-of-sequence token
    decoder.config.eos_token_id = tokenizer.sep_token_id  # Assuming sep_token_id is the end-of-sequence token
    decoder.config.pad_token_id = tokenizer.pad_token_id  # Padding token index
    return decoder


def vietocr(path_weight: str, device: str, type = 'v1'):
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = path_weight
    config['cnn']['pretrained']=False
    config['device'] = device
    config['predictor']['beamsearch']=False
    if type == 'v2':
        # config['transformer']['num_decoder_layers'] = 10
        # config['transformer']['nhead'] = 16
        HanNom_vocab = open('/home1/vudinh/NomNaOCR/HanNom_Vocab/unique_vocab.txt').read().splitlines()
    elif type == 'v3':
        HanNom_vocab = open('/home1/vudinh/NomNaOCR/HanNom_Vocab/unique_vocab.txt').read().splitlines()
        HanNom_vocab = open('/home1/vudinh/NomNaOCR/HanNom_Vocab/HanNom_vocab.txt').read().splitlines()
        config['transformer']['d_model'] = 512
    else:
        HanNom_vocab = open('/home1/vudinh/NomNaOCR/PaddleOCR-release-2.6/dataset/detection/Patches/CLC_and_Synthesis_vocab.txt').read().splitlines()
    model, vocab = build_model(config, HanNom_vocab)
    weights = config['weights']
    if weights != '':
        model.load_state_dict(torch.load(weights, map_location=torch.device(device)))
    return model
def trocr(vocab, tokenizer):
    model = VisionEncoderDecoderModel.from_pretrained('/home1/vudinh/NomNaOCR/weights/Accident_Chinese_doc/kotenseki-trocr-honkoku-ver2')
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.decoder.vocab_size = len(tokenizer)
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.max_length = 100
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    print(model.decoder)
    assert False

class UnionVietOCR(nn.Module):
    def __init__(self,
                 device="cuda",
                 embed_dim_vit=384,
                 max_length_token=128,
                 img_width=128,
                 img_height=32,
                 patch_size=4,
                 vietocr_pretrained='',
                 decoder_vietocr_pretrained='',
                 fc_vietocr_pretrained='',
                 union_pretrained='/home1/vudinh/NomNaOCR/icocr/original_pretrain/maerec_b_union14m.pth',
                 vocab_leng = 233,
                 type = 'v1'
                 ):
        super().__init__()

        self.device = device
        ##union vit
        img_size = (img_height,img_width)
        self.vit_model = VisionTransformer(img_size=img_size, pretrained=None, embed_dim=embed_dim_vit, patch_size=patch_size)
        # if img_size == (32,128) and union_pretrained != '':
        if True:
            # print("[INFO] Load pretrained maerec Union14M")
            try:
                checkpoint = torch.load(union_pretrained)["state_dict"]
                checkpoint = {k.replace('backbone.', ''):v for k,v in checkpoint.items()}
                # del checkpoint['pos_embed']
                self.vit_model.load_state_dict(checkpoint, strict =False)
                # assert False
            except:
                pass

        ##mapping enc && dec
        if type != 'v3':
            self.d_model = 256
        else: self.d_model = 512
        self.enc_to_dec_proj = nn.Linear(embed_dim_vit, self.d_model)
        self.embed_tgt = nn.Embedding(vocab_leng + 5, self.d_model)
        self.pos_enc = PositionalEncoding(self.d_model, 0.1, max_length_token)

        ##vietocr decoder
        vietocr_model = vietocr(vietocr_pretrained, device=self.device, type = type)
        self.decoder = vietocr_model.transformer.transformer.decoder
        if decoder_vietocr_pretrained != '':
            self.decoder.load_state_dict(torch.load(decoder_vietocr_pretrained, map_location=torch.device(self.device)))

        ##vietocr fc
        self.fc = vietocr_model.transformer.fc 
        if fc_vietocr_pretrained != '':
            self.fc.load_state_dict(torch.load(fc_vietocr_pretrained, map_location=torch.device(self.device)))
    def forward(self,
                img: torch.Tensor,
                tgt_input=None,
                tgt_padding_mask=None,
                tgt_output = None):
        encoder_output = self.vit_model(img)
        encoder_output = self.enc_to_dec_proj(encoder_output)
        encoder_output = encoder_output.transpose(1,0)

        tgt = self.pos_enc(self.embed_tgt(tgt_input) * math.sqrt(self.d_model))
        tgt_mask = self.gen_nopeek_mask(tgt_input.shape[0]).to(self.device)

        decoder_outputs = self.decoder(
            tgt=tgt,
            memory=encoder_output,
            tgt_mask = tgt_mask, 
            tgt_key_padding_mask = tgt_padding_mask
        )
        decoder_outputs= decoder_outputs.transpose(0, 1)
        decoder_outputs = self.fc(decoder_outputs)
        return decoder_outputs


    def gen_nopeek_mask(self, length):
        mask = (torch.triu(torch.ones(length, length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward_decoder(self, tgt, memory):
        tgt_mask = self.gen_nopeek_mask(tgt.shape[0]).to(tgt.device)
        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))
        # tgt = tgt.transpose(1,0)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        output = output.transpose(0, 1)
        return self.fc(output), memory
    

    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=256):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]

        return self.dropout(x)
class UnionRoBerta(nn.Module):
    def __init__(self,
                 device="cuda",
                 embed_dim_vit=384,
                 max_length_token=128,
                 img_width=128,
                 img_height=32,
                 patch_size=4,
                 vietocr_pretrained='',
                 decoder_vietocr_pretrained='',
                 fc_vietocr_pretrained='',
                 union_pretrained='/home1/vudinh/NomNaOCR/icocr/original_pretrain/maerec_b_union14m.pth',
                 type = 'v1',
                 tokenizer = None, 
                 huggingface_repo = "KoichiYasuoka/roberta-base-japanese-aozora-char"
                 ):
        super().__init__()
        """   
        The [`RobertaForCausalLM`] forward method, overrides the `__call__` special method.

    <Tip>

    Although the recipe for forward pass needs to be defined within this function, one should call the [`Module`]
    instance afterwards instead of this since the former takes care of running the pre and post processing steps while
    the latter silently ignores them.

    </Tip>

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`RobertaTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.

        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).


        Returns:
            [`transformers.modeling_outputs.CausalLMOutputWithCrossAttentions`] or `tuple(torch.FloatTensor)`: A [`transformers.modeling_outputs.CausalLMOutputWithCrossAttentions`] or a tuple of
            `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
            elements depending on the configuration ([`RobertaConfig`]) and inputs.

            - **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
            - **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            - **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
              shape `(batch_size, sequence_length, hidden_size)`.

              Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            - **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
              sequence_length)`.

              Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
              heads.
            - **cross_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
              sequence_length)`.

              Cross attentions weights after the attention softmax, used to compute the weighted average in the
              cross-attention heads.
            - **past_key_values** (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- Tuple of `torch.FloatTensor` tuples of length `config.n_layers`, with each tuple containing the cached key,
              value states of the self-attention and the cross-attention layers if model is used in encoder-decoder
              setting. Only relevant if `config.is_decoder = True`.

              Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
              `past_key_values` input) to speed up sequential decoding.
            """

        self.device = device
        ##union vit
        img_size = (img_height,img_width)
        self.vit_model = VisionTransformer(img_size=img_size, pretrained=None, embed_dim=embed_dim_vit, patch_size=patch_size)
        if img_size == (32,128) and union_pretrained != '':
            # print("[INFO] Load pretrained maerec Union14M")
            try:
                checkpoint = torch.load(union_pretrained)["state_dict"]
                checkpoint = {k.replace('backbone.', ''):v for k,v in checkpoint.items()}
                self.vit_model.load_state_dict(checkpoint, strict =False)
            except Exception as e:
                pass

        ##mapping enc && dec
        self.d_model = 768
        self.enc_to_dec_proj = nn.Linear(embed_dim_vit, self.d_model)
        
        ##vietocr decoder
        self.decoder = hugging_face_decoder_init(huggingface_repo, tokenizer)
        self.tokenizer = tokenizer
    def forward(self,
                img: torch.Tensor,
                tgt_input=None,
                tgt_padding_mask=None,
                tgt_output = None):
        encoder_output = self.vit_model(img)
        encoder_output = self.enc_to_dec_proj(encoder_output)
        decoder_outputs = self.decoder(
            input_ids=tgt_input,
            attention_mask=tgt_padding_mask,
            encoder_hidden_states=encoder_output,
            labels=tgt_output,
            output_attentions=False  # Enable attention outputs
        )
        return decoder_outputs.loss, decoder_outputs.logits


    def string_accuracy(self, predicts, labels):
        # Pad the shorter string with spaces to match the lengths
        acc = 0
        for s1, s2 in zip(predicts, labels):
            max_len = max(len(s1), len(s2))
            s1 = s1.ljust(max_len)
            s2 = s2.ljust(max_len)
            # Calculate the percentage of matching characters
            matches = sum(c1 == c2 for c1, c2 in zip(s1, s2))/max_len
            acc += matches
        return acc/len(predicts)
    def inference(self, img: torch.Tensor, tgt_output: torch.Tensor = None, max_seq_len: int = 128, inference = False):
        """
        Perform inference with the model to generate text predictions for the given input image.
        
        Args:
            img (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).
            max_seq_len (int): Maximum sequence length for generated tokens. Default is 128.
        
        Returns:
            List[str]: A list of decoded text predictions for each image in the batch.
        """
        self.eval()  # Set the model to evaluation mode
        if not inference:
            tgt_output.masked_fill_(tgt_output == -100, -self.tokenizer.pad_token_id)
            labels = [self.tokenizer.decode(list(seq), skip_special_tokens=True) for seq in tgt_output]
        list_cross_attention = []
        with torch.no_grad():
            # Encode the image using the Vision Transformer
            encoder_output = self.vit_model(img)
            encoder_output = self.enc_to_dec_proj(encoder_output)
            
            
            # Initialize decoder input with the start token
            batch_size = img.size(0)
            start_token_id = self.tokenizer.cls_token_id  # Start token ID
            end_token_id = self.tokenizer.sep_token_id   # End token ID
            
            tgt_input = torch.full(
                (batch_size, 1), start_token_id, dtype=torch.long, device=img.device
            )
            
            # Generate tokens iteratively
            for _ in range(max_seq_len):
                decoder_outputs = self.decoder(
                    input_ids=tgt_input,
                    encoder_hidden_states=encoder_output,
                    output_attentions=True
                )
                logits = decoder_outputs.logits[:, -1, :]  # Shape: (batch_size, vocab_size)
                next_token = logits.argmax(dim=-1, keepdim=True)  # Get token with highest probability
                
                # Append the predicted token
                tgt_input = torch.cat([tgt_input, next_token], dim=1)
                # Stop generation if all sequences predict the end token
                if (next_token == end_token_id).all():
                    break
                
            # Decode the generated token sequences
            decoded_texts = [
                self.tokenizer.decode(seq, skip_special_tokens=True) for seq in tgt_input
            ]
        # visualize_cross_attention(decoded_texts, decoder_outputs.cross_attentions)
        # assert False
        if inference: 
            return decoded_texts
        return decoded_texts, labels, self.string_accuracy(decoded_texts, labels)
class PatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.

    """

    def __init__(self, image_size=224, patch_size=16, num_channels=3, embed_dim=768):
        super().__init__()
        image_size = self.to_2tuple(image_size)
        patch_size = self.to_2tuple(patch_size)
        
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    def to_2tuple(self,x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return (x, x)

    def forward(self, pixel_values, interpolate_pos_encoding=False):
        batch_size, num_channels, height, width = pixel_values.shape
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x

 
class ModifiedViTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = PatchEmbeddings(
            image_size=(512, 48),  # Updated image size
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            embed_dim=config.hidden_size,
        )
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def interpolate_pos_encoding(self, embeddings, height, width):
        npatch = embeddings.shape[1] - 1
        N = self.position_embeddings.shape[1] - 1
        if npatch == N and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, pixel_values, interpolate_pos_encoding=False):
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings
        return self.dropout(embeddings)


class TrRoBerta(nn.Module):
    def __init__(self,
                 device="cuda",
                 embed_dim_vit=384,
                 max_length_token=128,
                 img_width=128,
                 img_height=32,
                 patch_size=4,
                 vietocr_pretrained='',
                 decoder_vietocr_pretrained='',
                 fc_vietocr_pretrained='',
                 encoder_trocr_pretrained='/home1/vudinh/NomNaOCR/weights/Accident_Chinese_doc/kotenseki-trocr-honkoku-ver2',
                 type = 'v1',
                 tokenizer = None, 
                 huggingface_repo = "KoichiYasuoka/roberta-base-japanese-aozora-char"
                 ):
        super().__init__()

        self.device = device
        ##union vit
        img_size = (img_height,img_width)
        self.vit_model = VisionEncoderDecoderModel.from_pretrained(encoder_trocr_pretrained).encoder
        
        ##vietocr decoder
        self.decoder = hugging_face_decoder_init(huggingface_repo, tokenizer)
        self.tokenizer = tokenizer
    def forward(self,
                img: torch.Tensor,
                tgt_input=None,
                tgt_padding_mask=None,
                tgt_output = None):
        encoder_output = self.vit_model(img).last_hidden_state
        # encoder_output = self.enc_to_dec_proj(encoder_output)
        decoder_outputs = self.decoder(
            input_ids=tgt_input,
            attention_mask=tgt_padding_mask,
            encoder_hidden_states=encoder_output,
            labels=tgt_output,
            output_attentions=True  # Enable attention outputs
        )
        return decoder_outputs.loss, decoder_outputs.logits


    def string_accuracy(self, predicts, labels):
        # Pad the shorter string with spaces to match the lengths
        acc = 0
        for s1, s2 in zip(predicts, labels):
            max_len = max(len(s1), len(s2))
            s1 = s1.ljust(max_len)
            s2 = s2.ljust(max_len)
            # Calculate the percentage of matching characters
            matches = sum(c1 == c2 for c1, c2 in zip(s1, s2))/max_len
            acc += matches
        return acc/len(predicts)
    def inference(self, img: torch.Tensor, tgt_output: torch.Tensor = None, max_seq_len: int = 128, inference = False):
        """
        Perform inference with the model to generate text predictions for the given input image.
        
        Args:
            img (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).
            max_seq_len (int): Maximum sequence length for generated tokens. Default is 128.
        
        Returns:
            List[str]: A list of decoded text predictions for each image in the batch.
        """
        self.eval()  # Set the model to evaluation mode
        if not inference:
            tgt_output.masked_fill_(tgt_output == -100, -self.tokenizer.pad_token_id)
            labels = [self.tokenizer.decode(list(seq), skip_special_tokens=True) for seq in tgt_output]
        list_cross_attention = []
        with torch.no_grad():
            # Encode the image using the Vision Transformer
            encoder_output = self.vit_model(img).last_hidden_state
            # encoder_output = self.enc_to_dec_proj(encoder_output)
            
            
            # Initialize decoder input with the start token
            batch_size = img.size(0)
            start_token_id = self.tokenizer.cls_token_id  # Start token ID
            end_token_id = self.tokenizer.sep_token_id   # End token ID
            
            tgt_input = torch.full(
                (batch_size, 1), start_token_id, dtype=torch.long, device=img.device
            )
            
            # Generate tokens iteratively
            for _ in range(max_seq_len):
                decoder_outputs = self.decoder(
                    input_ids=tgt_input,
                    encoder_hidden_states=encoder_output,
                    output_attentions=True
                )
                logits = decoder_outputs.logits[:, -1, :]  # Shape: (batch_size, vocab_size)
                next_token = logits.argmax(dim=-1, keepdim=True)  # Get token with highest probability
                
                # Append the predicted token
                tgt_input = torch.cat([tgt_input, next_token], dim=1)
                # Stop generation if all sequences predict the end token
                if (next_token == end_token_id).all():
                    break
                
            # Decode the generated token sequences
            decoded_texts = [
                self.tokenizer.decode(seq, skip_special_tokens=True) for seq in tgt_input
            ]
        # visualize_cross_attention(decoded_texts, decoder_outputs.cross_attentions)
        # assert False
        if inference: 
            return decoded_texts
        return decoded_texts, labels, self.string_accuracy(decoded_texts, labels)
    
    
class TrRoBerta_custom(nn.Module):
    def __init__(self,
                 device="cuda",
                 embed_dim_vit=384,
                 max_length_token=128,
                 img_width=128,
                 img_height=32,
                 patch_size=4,
                 vietocr_pretrained='',
                 decoder_vietocr_pretrained='',
                 fc_vietocr_pretrained='',
                 encoder_trocr_pretrained='/home1/vudinh/NomNaOCR/weights/Accident_Chinese_doc/kotenseki-trocr-honkoku-ver2',
                 type = 'v1',
                 tokenizer = None, 
                 huggingface_repo = "KoichiYasuoka/roberta-base-japanese-aozora-char"
                 ):
        super().__init__()

        self.device = device
        ##union vit
        img_size = (img_height,img_width)
        self.vit_model = VisionEncoderDecoderModel.from_pretrained(encoder_trocr_pretrained).encoder
        config = self.vit_model.config
        # Modify the embeddings module
        config.image_size = (512, 48)
        modified_embeddings = ModifiedViTEmbeddings(config)
        self.vit_model.embeddings = modified_embeddings
        self.decoder = hugging_face_decoder_init(huggingface_repo, tokenizer)
        self.tokenizer = tokenizer
    def forward(self,
                img: torch.Tensor,
                tgt_input=None,
                tgt_padding_mask=None,
                tgt_output = None):
        encoder_output = self.vit_model(img).last_hidden_state
        decoder_outputs = self.decoder(
            input_ids=tgt_input,
            attention_mask=tgt_padding_mask,
            encoder_hidden_states=encoder_output,
            labels=tgt_output,
            output_attentions=True  # Enable attention outputs
        )
        return decoder_outputs.loss, decoder_outputs.logits


    def string_accuracy(self, predicts, labels):
        # Pad the shorter string with spaces to match the lengths
        acc = 0
        for s1, s2 in zip(predicts, labels):
            max_len = max(len(s1), len(s2))
            s1 = s1.ljust(max_len)
            s2 = s2.ljust(max_len)
            # Calculate the percentage of matching characters
            matches = sum(c1 == c2 for c1, c2 in zip(s1, s2))/max_len
            acc += matches
        return acc/len(predicts)
    def inference(self, img: torch.Tensor, tgt_output: torch.Tensor = None, max_seq_len: int = 128, inference = False):
        """
        Perform inference with the model to generate text predictions for the given input image.
        
        Args:
            img (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).
            max_seq_len (int): Maximum sequence length for generated tokens. Default is 128.
        
        Returns:
            List[str]: A list of decoded text predictions for each image in the batch.
        """
        self.eval()  # Set the model to evaluation mode
        if not inference:
            tgt_output.masked_fill_(tgt_output == -100, -self.tokenizer.pad_token_id)
            labels = [self.tokenizer.decode(list(seq), skip_special_tokens=True) for seq in tgt_output]
        list_cross_attention = []
        with torch.no_grad():
            # Encode the image using the Vision Transformer
            encoder_output = self.vit_model(img).last_hidden_state
            # encoder_output = self.enc_to_dec_proj(encoder_output)
            
            
            # Initialize decoder input with the start token
            batch_size = img.size(0)
            start_token_id = self.tokenizer.cls_token_id  # Start token ID
            end_token_id = self.tokenizer.sep_token_id   # End token ID
            
            tgt_input = torch.full(
                (batch_size, 1), start_token_id, dtype=torch.long, device=img.device
            )
            
            # Generate tokens iteratively
            for _ in range(max_seq_len):
                decoder_outputs = self.decoder(
                    input_ids=tgt_input,
                    encoder_hidden_states=encoder_output,
                    output_attentions=True
                )
                logits = decoder_outputs.logits[:, -1, :]  # Shape: (batch_size, vocab_size)
                next_token = logits.argmax(dim=-1, keepdim=True)  # Get token with highest probability
                
                # Append the predicted token
                tgt_input = torch.cat([tgt_input, next_token], dim=1)
                # Stop generation if all sequences predict the end token
                if (next_token == end_token_id).all():
                    break
                
            # Decode the generated token sequences
            decoded_texts = [
                self.tokenizer.decode(seq, skip_special_tokens=True) for seq in tgt_input
            ]
        # visualize_cross_attention(decoded_texts, decoder_outputs.cross_attentions)
        # assert False
        if inference: 
            return decoded_texts
        return decoded_texts, labels, self.string_accuracy(decoded_texts, labels)
    


def def_fun():
    image = torch.rand(1, 3, 32, 128).to("cuda")  # Example input image
    text_input = ["𬺚𬼀"]
    custom_vocab = HanNom_vocab = open('/home1/vudinh/NomNaOCR/HanNom_Vocab/unique_vocab.txt').read().splitlines()
    model = UnionRoBerta(device="cuda", custom_vocab=custom_vocab).to('cuda')
    output = model(image, tgt_input=text_input)
    print(output.shape)


