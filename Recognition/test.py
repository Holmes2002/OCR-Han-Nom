from union.mix_union_vietocr import UnionVietOCR, UnionRoBerta, UnionRoBerta, TrRoBerta_custom
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
from PIL import Image
from torchvision import transforms
from torch.nn.functional import cosine_similarity
from networkx import Graph
from networkx.algorithms.components import connected_components
from torch.utils.data import DataLoader, Dataset
import pickle
import shutil
config = Cfg.load_config_from_file("./config/trrobertaocr_512x48.yml")
vocab = config.vocab
device = config.device
HanNom_vocab = open('/home1/vudinh/NomNaOCR/HanNom_Vocab/unique_vocab_and_opensource.txt').read().splitlines()
# HanNom_vocab = open('vocab_tmp.txt').read().splitlines()
tokenizer = AutoTokenizer.from_pretrained("KoichiYasuoka/roberta-small-japanese-aozora-char")
tokenizer = tokenizer.train_new_from_iterator(HanNom_vocab, vocab_size=len(HanNom_vocab))
os.makedirs(config.ckpt_save_path, exist_ok = True)
log_file = open(f'{config.ckpt_save_path}/log_train.txt', 'w')
model = TrRoBerta_custom(max_length_token=config.max_length_token,
                     img_width=config.width_size,
                     img_height=config.height_size,
                     patch_size=config.patch_size,
                     tokenizer = tokenizer,
                     type = 'v2',
                     embed_dim_vit=config.embed_dim_vit
                     )
if config.ckpt != '':
    model.load_state_dict(torch.load(config.ckpt))
root = '/home1/vudinh/NomNaOCR/Handwritten-CycleGAN/SVG2PNG'
model.load_state_dict(torch.load("/home1/vudinh/NomNaOCR/icocr/trocr/pretrain_512x48/epoch_2.pth"), strict = False)
BATCH_SIZE = 64
SIMILARITY_THRESHOLD = 0.85
input_dir = '/home1/vudinh/NomNaOCR/Handwritten-CycleGAN/SVG2PNG'
output_dir = '/home1/vudinh/clusters'
embedding_save_path = '/home1/vudinh/clusters/embeddings.pkl'
os.makedirs(output_dir, exist_ok=True)

# Custom Dataset
class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.image_paths = [os.path.join(root, f) for f in os.listdir(root)]
        self.transform = transform or transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).rotate(90, expand=True).resize((512, 48)).convert("RGB")
        return self.transform(image), image_path

# Load dataset and dataloader
dataset = ImageDataset(input_dir, transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
model.load_state_dict(torch.load("/home1/vudinh/NomNaOCR/icocr/trocr/pretrain_512x48/epoch_2.pth"), strict=False)

# Extract embeddings
all_embeddings = []
all_image_paths = []

# model.eval()
# with torch.no_grad():
#     for batch_images, batch_paths in tqdm(dataloader, desc="Extracting embeddings"):
#         # Forward pass through the model
#         batch_embeddings = model.vit_model(batch_images).last_hidden_state
#         # Average over sequence length (dim=1)
#         batch_embeddings = batch_embeddings.mean(dim=1)  # Shape: [batch_size, 768]
#         all_embeddings.append(batch_embeddings)
#         all_image_paths.extend(batch_paths)

# # Combine embeddings into a single tensor
# embeddings_tensor = torch.cat(all_embeddings, dim=0)  # Shape: [num_images, 768]
BATCH_SIZE = 64
SIMILARITY_THRESHOLD = 0.85
CHUNK_SIZE = 500  # Chunk size for similarity calculation
input_dir = '/home1/vudinh/NomNaOCR/Handwritten-CycleGAN/SVG2PNG'
output_dir = '/home1/vudinh/clusters'
os.makedirs(output_dir, exist_ok=True)

# # Save embeddings and paths
# with open(embedding_save_path, 'wb') as f:
#     pickle.dump({"embeddings": embeddings_tensor, "paths": all_image_paths}, f)
with open(embedding_save_path, 'rb') as f:
    data = pickle.load(f)
embeddings_tensor = data["embeddings"]
all_image_paths = data["paths"]

print(f"Embeddings saved at: {embedding_save_path}")

# Function to compute pairwise similarities in chunks
def compute_similarity_in_chunks(embeddings, chunk_size=500):
    num_images = embeddings.size(0)
    similarity_matrix = torch.zeros((num_images, num_images))  # Placeholder for the full matrix

    for i in tqdm(range(0, num_images, chunk_size), desc="Chunk-wise similarity calculation"):
        end_i = min(i + chunk_size, num_images)
        chunk_embeddings_i = embeddings[i:end_i]  # Select chunk i

        for j in range(0, num_images, chunk_size):
            end_j = min(j + chunk_size, num_images)
            chunk_embeddings_j = embeddings[j:end_j]  # Select chunk j

            # Compute cosine similarity for the chunk
            chunk_similarity = cosine_similarity(
                chunk_embeddings_i.unsqueeze(1),  # Shape: [chunk_size_i, 1, embedding_dim]
                chunk_embeddings_j.unsqueeze(0),  # Shape: [1, chunk_size_j, embedding_dim]
                dim=-1
            )
            # Update the corresponding part of the similarity matrix
            similarity_matrix[i:end_i, j:end_j] = chunk_similarity

    return similarity_matrix

# Compute pairwise similarity matrix
print("Computing similarity matrix...")
similarity_matrix = compute_similarity_in_chunks(embeddings_tensor, CHUNK_SIZE)

# Build a graph of similar images
print("Building graph of similar images...")
graph = Graph()
num_images = embeddings_tensor.size(0)
graph.add_nodes_from(range(num_images))  # Add one node for each image

# Add edges for similar images
for i in tqdm(range(num_images), desc="Adding edges to graph"):
    for j in range(i + 1, num_images):
        if similarity_matrix[i, j] > SIMILARITY_THRESHOLD:
            graph.add_edge(i, j)

# Find clusters (connected components in the graph)
print("Finding connected components (clusters)...")
clusters = list(connected_components(graph))

# Save clustered images
print("Saving clusters...")
for cluster_idx, cluster in enumerate(clusters):
    cluster_dir = os.path.join(output_dir, f"cluster_{cluster_idx}")
    os.makedirs(cluster_dir, exist_ok=True)
    for image_idx in cluster:
        src_path = all_image_paths[image_idx]
        dest_path = os.path.join(cluster_dir, os.path.basename(src_path))
        shutil.copy(src_path, dest_path)

print(f"Clustering complete! Results saved in: {output_dir}")
