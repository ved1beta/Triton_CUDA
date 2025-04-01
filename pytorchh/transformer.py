import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections and reshape for multi-head attention
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape back to original dimensions
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.w_o(context)
        
        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to input
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Feed forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Cross-attention with residual connection and layer norm
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = x + self.dropout2(attn_output)
        x = self.norm2(x)
        
        # Feed forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = x + self.dropout3(ff_output)
        x = self.norm3(x)
        
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_length=5000, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # Embed tokens and add positional encoding
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.positional_encoding(x)
        
        # Process through encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Apply final layer norm
        x = self.norm(x)
        
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_length=5000, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Embed tokens and add positional encoding
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.positional_encoding(x)
        
        # Process through decoder layers
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        
        # Apply final layer norm
        x = self.norm(x)
        
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, d_ff=2048, 
                 num_enc_layers=6, num_dec_layers=6, max_seq_length=5000, dropout=0.1):
        super().__init__()
        
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_enc_layers, max_seq_length, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_dec_layers, max_seq_length, dropout)
        
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode source sequence
        enc_output = self.encoder(src, src_mask)
        
        # Decode target sequence
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        
        # Project to vocabulary
        output = self.output_projection(dec_output)
        
        return output
    
    @staticmethod
    def create_padding_mask(seq, pad_idx=0):
        """Create mask for padding tokens"""
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    
    @staticmethod
    def create_look_ahead_mask(size):
        """Create mask to prevent attending to future tokens"""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return ~mask


# Example of creating the model and a forward pass
def transformer_example():
    # Model parameters
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    dropout = 0.1
    
    # Create model
    transformer = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_enc_layers=num_layers,
        num_dec_layers=num_layers,
        dropout=dropout
    )
    
    # Sample data
    batch_size = 32
    src_seq_len = 50
    tgt_seq_len = 40
    
    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    # Create masks
    src_pad_mask = Transformer.create_padding_mask(src)
    tgt_pad_mask = Transformer.create_padding_mask(tgt)
    tgt_look_ahead_mask = Transformer.create_look_ahead_mask(tgt_seq_len)
    
    # Combine masks for decoder
    tgt_mask = tgt_pad_mask & tgt_look_ahead_mask
    
    # Forward pass
    output = transformer(src, tgt, src_pad_mask, tgt_mask)
    
    return output


# Example of training loop
def train_transformer(model, train_dataloader, optimizer, criterion, num_epochs):
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch_idx, (src, tgt) in enumerate(train_dataloader):
            # Prepare input and target
            input_tgt = tgt[:, :-1]  # Remove last token (shifted right)
            target_tgt = tgt[:, 1:]  # Remove first token (shifted left)
            
            # Create masks
            src_mask = Transformer.create_padding_mask(src)
            tgt_pad_mask = Transformer.create_padding_mask(input_tgt)
            tgt_look_ahead_mask = Transformer.create_look_ahead_mask(input_tgt.size(1))
            tgt_mask = tgt_pad_mask & tgt_look_ahead_mask
            
            # Forward pass
            optimizer.zero_grad()
            output = model(src, input_tgt, src_mask, tgt_mask)
            
            # Reshape for loss calculation
            output = output.contiguous().view(-1, output.size(-1))
            target_tgt = target_tgt.contiguous().view(-1)
            
            # Calculate loss
            loss = criterion(output, target_tgt)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch: {epoch+1}, Average Loss: {avg_loss:.4f}")