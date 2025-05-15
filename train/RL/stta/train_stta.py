import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import pandas as pd
import networkx as nx
from scipy.sparse.linalg import eigsh
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, queries, features, answers, tokenizer):
        self.queries = queries
        self.features = features
        self.answers = answers
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.features[idx], self.answers[idx]


class STTA(nn.Module):
    def __init__(self, feature_dim, hidden_dims, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(feature_dim, hidden_dims[0])
        

        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dims[0], num_heads=num_heads, dropout=dropout)
        

        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.GELU(),
                nn.LayerNorm(hidden_dims[i+1]),
                nn.Dropout(dropout)
            ])
        self.mlp_blocks = nn.Sequential(*layers)
        

        self.output_proj = nn.Linear(hidden_dims[-1], embed_dim)
        

        self.residual_norm = nn.LayerNorm(hidden_dims[0])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, feature):

        x = self.input_proj(feature)
        x = self.dropout(x)
        

        x = x.unsqueeze(0)
        attn_output, _ = self.self_attn(x, x, x)
        x = self.residual_norm(x + attn_output)
        x = x.squeeze(0)
        

        x = self.mlp_blocks(x)
        

        x = self.output_proj(x)
        return x


def compute_laplacian_eigenvalues(graph, M):
    try:
        L = nx.normalized_laplacian_matrix(graph).astype(np.float32)
        eigenvalues, _ = eigsh(L, k=M, which='SM')
        return torch.tensor(eigenvalues, dtype=torch.float32)
    except Exception as e:
        print(f"Error computing eigenvalues: {e}")
        return torch.zeros(M, dtype=torch.float32)

def train(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    

    for param in model.parameters():
        param.requires_grad = False


    feature_block = STTA(
        feature_dim=args.feature_dim,
        hidden_dims=args.hidden_dims,
        embed_dim=model.config.hidden_size,
        num_heads=args.num_heads,
        dropout=args.dropout
    )
    

    optimizer = torch.optim.Adam(feature_block.parameters(), lr=args.lr)
    

    try:
        df = pd.read_csv(args.csv_file)
        queries = df['querys'].tolist()
        answers = df['answers'].tolist()
        if len(queries) != len(answers):
            raise ValueError("Number of queries and answers must match")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return


    try:
        graph = nx.read_gexf(args.gexf_file)
        eigenvalues = compute_laplacian_eigenvalues(graph, args.feature_dim)
        features = [eigenvalues for _ in range(len(queries))]
    except Exception as e:
        print(f"Error processing GEXF file: {e}")
        features = [torch.zeros(args.feature_dim) for _ in range(len(queries))]


    dataset = CustomDataset(queries, features, answers, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    model.to(device)
    feature_mlp.to(device)


    model.eval()
    feature_mlp.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in dataloader:
            query, feature, answer = batch
            

            inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            embeddings = model.get_input_embeddings()(input_ids)
            

            feature = torch.stack([f.to(device) for f in feature])
            feature_embed = feature_block(feature)
            feature_embed = feature_embed.unsqueeze(1)
            

            fused_embeddings = torch.cat([feature_embed, embeddings], dim=1)
            feature_mask = torch.ones(fused_embeddings.size(0), 1, device=device)
            fused_attention_mask = torch.cat([feature_mask, attention_mask], dim=1)
            

            target_inputs = tokenizer(answer, return_tensors="pt", padding=True, truncation=True, max_length=128)
            target_ids = target_inputs["input_ids"].to(device)
            

            outputs = model(inputs_embeds=fused_embeddings, attention_mask=fused_attention_mask)
            logits = outputs.logits[:, 1:, :]
            

            loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            loss = loss_fn(logits.reshape(-1, model.config.vocab_size), target_ids.reshape(-1))
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}")
        

        if (epoch + 1) % args.save_interval == 0:
            torch.save(feature_block.state_dict(), os.path.join(args.save_dir, f"feature_block_epoch_{epoch+1}.pth"))
            print(f"Saved model at epoch {epoch+1}")

def main():
    parser = argparse.ArgumentParser(description="Train STTA with Qwen2.5-7B and Laplacian Eigenvalues")
    parser.add_argument("--model_name", type=str, default="models/Qwen2.5-7B-Instruct", help="Pretrained model name")
    parser.add_argument("--feature_dim", type=int, default=10, help="Number of Laplacian eigenvalues (M)")
    parser.add_argument("--hidden_dims", type=int, nargs='+', default=[256, 128, 64], help="hidden dimensions")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save models")
    parser.add_argument("--save_interval", type=int, default=2, help="Save model every N epochs")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV file with queries and answers")
    parser.add_argument("--gexf_file", type=str, required=True, help="Path to GEXF file with graph")
    
    args = parser.parse_args()
    

    os.makedirs(args.save_dir, exist_ok=True)
    
    train(args)

if __name__ == "__main__":
    main()