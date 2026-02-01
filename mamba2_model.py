import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Import the Mamba2 class
from mamba2 import Mamba2


class Mamba2Model(nn.Module):
    """
    Mamba2-based model for T-PRIME protocol classification.
    This model replaces the transformer architecture with Mamba2 SSM layers.
    """
    
    def __init__(self, classes=4, d_model=256, seq_len=64, nlayers=2, 
                 d_state=128, d_conv=4, expand=2, headdim=64, 
                 d_ssm=None, ssm_ratio=0.8, ngroups=1, A_init_range=(1, 16),
                 rmsnorm=True, norm_before_gate=False, chunk_size=256):
        super().__init__()
        
        self.classes = classes
        self.d_model = d_model
        self.seq_len = seq_len
        self.nlayers = nlayers
        
        # Input projection will be created dynamically based on input size
        self.input_proj = None
        
        # Determine how many inner channels use SSM vs gated MLP
        # d_inner = expand * d_model; enforce d_ssm multiple of headdim
        d_inner = expand * d_model
        if d_ssm is None:
            target = int(round(d_inner * float(ssm_ratio)))
            # round to nearest multiple of headdim within [headdim, d_inner]
            target = max(headdim, min(d_inner, (target // headdim) * headdim))
            d_ssm_final = target
        else:
            d_ssm_final = int(d_ssm)
            # clamp and align to headdim
            d_ssm_final = max(headdim, min(d_inner, (d_ssm_final // headdim) * headdim))

        # Stack of Mamba2 layers (pure Mamba2, no attention)
        self.mamba_layers = nn.ModuleList([
            Mamba2(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
                d_ssm=d_ssm_final,
                ngroups=ngroups,
                A_init_range=A_init_range,
                rmsnorm=rmsnorm,
                norm_before_gate=norm_before_gate,
                chunk_size=chunk_size,
                layer_idx=i
            ) for i in range(nlayers)
        ])
        
        # Final classification head (no attention pooling)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, classes),
            nn.LogSoftmax(dim=-1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the model."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass through the Mamba2 model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model) or (batch_size, 2*slice_length)
        
        Returns:
            Output logits of shape (batch_size, classes)
        """
        # Handle input format: if 1D (2*slice_length), reshape to 2D (seq_len, d_model)
        if x.dim() == 2:
            # Input is (batch_size, 2*slice_length)
            batch_size, seq_len_flat = x.shape
            
            # Calculate the feature dimension that fits the sequence length
            # We want: batch_size * seq_len * d_model = seq_len_flat
            # So: d_model = seq_len_flat / seq_len
            if seq_len_flat % self.seq_len != 0:
                # If it doesn't divide evenly, pad or truncate to make it fit
                target_size = self.seq_len * self.d_model
                if seq_len_flat < target_size:
                    # Pad with zeros
                    padding = target_size - seq_len_flat
                    x = torch.cat([x, torch.zeros(batch_size, padding, device=x.device)], dim=1)
                else:
                    # Truncate
                    x = x[:, :target_size]
                seq_len_flat = target_size
            
            # Now reshape to (batch_size, seq_len, d_model)
            actual_d_model = seq_len_flat // self.seq_len
            x = x.view(batch_size, self.seq_len, actual_d_model)
            
            # Create input projection if it doesn't exist or if dimensions changed
            if self.input_proj is None or self.input_proj.in_features != actual_d_model:
                self.input_proj = nn.Linear(actual_d_model, self.d_model).to(x.device)
        else:
            # Input is already 3D (batch_size, seq_len, d_model)
            actual_d_model = x.shape[2]
            if self.input_proj is None or self.input_proj.in_features != actual_d_model:
                self.input_proj = nn.Linear(actual_d_model, self.d_model).to(x.device)
        
        # Ensure contiguous memory layout
        x = x.contiguous()

        # Input projection
        x = self.input_proj(x)
        
        # Pass through Mamba2 layers (no extra logic)
        for mamba_layer in self.mamba_layers:
            x = mamba_layer(x)
        
        # Global average pooling over sequence dimension
        x = torch.mean(x, dim=1)  # (batch_size, d_model)
        
        # Classification
        x = self.classifier(x)
        
        return x

"""
Mamba2Model is intentionally minimal:
- dynamic input projection to match incoming feature dim
- sequential stack of Mamba2 blocks
- mean pooling over time
- classifier head with LogSoftmax
No attention, no residual extras, no auxiliary pooling.
"""
