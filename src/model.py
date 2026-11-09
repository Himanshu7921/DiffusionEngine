import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

class NoiseModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor, t_e: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class TimeStepEmbeddings(nn.Module):
    def __init__(self, time_embed: int):
        super().__init__()
        self.time_embed = time_embed # assumes an even embedding dim
        self.half_dim = self.time_embed // 2

        i = torch.arange(0, self.half_dim)
        denominator = (10000 ** (2*i / self.time_embed))
        self.register_buffer("denominator", denominator)

    def forward(self, t: torch.Tensor):
        """
        expects 't': (B,)
        returs t_e: (B, d); d: timestep embedding dim (saved in config)
        """
        t = t.float().unsqueeze(-1) # (B) -> (B, 1)
        sinusoidal_input = t / self.denominator.unsqueeze(0) # (B, 1) / (1, d/2) --> (B, d/2)
        t_e = torch.cat([torch.sin(sinusoidal_input), torch.cos(sinusoidal_input)], dim = -1) # (B, d/2) + (B, d/2) -> (B, d)
        return t_e

class DownSampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        performs one Down sampling operation, it's a part of UNet Model
        """
        super().__init__()
        self.downsample = nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                stride = 2,
                kernel_size = 3,
                padding = 1
            )

    def forward(self, x_t: torch.Tensor):
        """
        x_t: (B, C_in, H_in, W_in) (Noise Image, noise injected upto timestep 't'; )
        returns: (B, C_out, H_out, W_out)
        """
        return self.downsample(x_t)
    
class UpSampleBlock(nn.Module):
    """
    performs one Up sampling operation, it's a part of UNet Model
    in-order to make the implementation use any input spatial dim, i'll implement interpolation (nearest neighbour)
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                stride = 1,
                kernel_size = 3,
                padding = 1
            )

    def forward(self, x_t: torch.Tensor, target_size: int = None):
        """
        x_t: (B, C_in, H_in, W_in) (Noise Image, noise injected upto timestep 't'; )
        returns: (B, C_out, H_out, W_out)
        """
        if target_size is not None:
            x_t = F.interpolate(
                x_t, size = target_size, mode = "nearest"
            )
        else:
            x_t = F.interpolate(
                x_t, scale_factor = 2, mode = "nearest"
            )
        return self.upsample(x_t)

class ResBlock(nn.Module):
    """
    Residual block with:
        GroupNorm -> SiLU -> Conv
        Timestep projection (SiLU -> Linear)
        GroupNorm -> SiLU -> Conv
        Residual connection
    """
    def __init__(self, in_channels: int, out_channels: int, time_embd_dim: int, groups: int = 8):
        super().__init__()
        self.group_norm_1 = nn.GroupNorm(num_groups = groups, num_channels = in_channels)
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, padding = 1, stride = 1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embd_dim, out_features = out_channels)
        )

        self.group_norm_2 = nn.GroupNorm(num_groups = groups, num_channels = out_channels)
        self.conv2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, padding = 1, stride = 1)

        self.silu = nn.SiLU()

        # make sure spatial-dims matches, because we need to apply 'residual-skip connections'
        if in_channels != out_channels:
            self.fn = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.fn = nn.Identity()

        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x_t: torch.Tensor, t_e: torch.Tensor):
        """
        x_t: [B, C, H, W]
        t_e: [B, D]; D = time_embed_dim

        returns: (B, C_out, H, W)
        """

        h = self.conv1(self.silu(self.group_norm_1(x_t))) # (B, C_in, H, W) --> (B, C_out, H, W)
        t_e = self.time_proj(t_e).unsqueeze(-1).unsqueeze(-1) # (B, D) --> (B, C_out, 1, 1)

        h = h + t_e # (B, C_out, H, W) + (B, C_out, 1, 1) --> (B, C_out, H, W)

        h = self.conv2(self.silu(self.group_norm_2(h))) # (B, C_out, H, W) --> (B, C_out, H, W)

        # apply residual-skip connections
        return h + self.fn(x_t) # (B, C_out, H, W) + (B, C_out, H, W)

class AttentionBlock(nn.Module):
    def __init__(self, channels: int, n_group: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(n_group, num_channels = channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

        self.out_proj = nn.Linear(channels, channels)

    def forward(self, x_t: torch.Tensor):
        """
        x.shape = (B, C, H, W)

        returns (B, C, H, W) such that it "mixes the information globally"
        """
        x = self.norm(x_t)

        B, C, H, W = x.shape
        # (B, C, H, W) --> (B, H, W, C) | (0, 1, 2, 3) --> (0, 2, 3, 1)

        Q = self.q(x).permute(0, 2, 3, 1).reshape(B, H*W, C) # (B, C, H, W) --> (B, H*W, C)
        K = self.k(x).permute(0, 2, 3, 1).reshape(B, H*W, C) # (B, C, H, W) --> (B, H*W, C)
        V = self.v(x).permute(0, 2, 3, 1).reshape(B, H*W, C) # (B, C, H, W) --> (B, H*W, C)
        A = F.softmax((Q @ K.transpose(-1, -2)) * self.scale, dim = -1) # (B, H*W, C) @ (B, C, H*W) --> (B, H*W, H*W)
        scores = A @ V # (B, H*W, H*W) @ (B, H*W, C) --> (B, H*W, C)
        out = self.out_proj(scores) # (B, H*W, C) --> (B, H*W, C)
        out = out.permute(0, 2, 1).reshape(B, C, H, W) # (B, H*W, C) --> (B, C, H*W) --> (B, C, H, W)
        return x_t + out

class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int, n_groups: int = 8):
        """
        responsibiloties:
        1. Extract meaningful spatial features using convolutional layers
        2. Injects the timestep information into the feature representations
        3. Refine the extracted features using residual connections
        4. save the feature maps as 'skip' that can be used to pass into the DecoderSage
        """
        super().__init__()

        self.resblock1 = ResBlock(in_channels, out_channels, time_embd_dim = time_embed_dim, groups = n_groups)
        self.resblock2 = ResBlock(out_channels, out_channels, time_embd_dim = time_embed_dim, groups = n_groups)

    def forward(self, x_t: torch.Tensor, t_e: torch.Tensor):
        """
        x_t: (B, C, H, W)
        t_e: (B, d)
        """
        x_t = self.resblock1(x_t, t_e) # (B, C, H, W) ---> (B, C_out, H, W)
        x_t = self.resblock2(x_t, t_e) # (B, C, H, W) ---> (B, C_out, H, W)
        return x_t

class DecoderStage(nn.Module):
    """
    responsibilities:
    1. Combine encoder and decoder features
    2. Process combined features through residual blocks
    3. Return reconstructed features for downstream layers
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, time_embed_dim: int, n_groups: int = 8):
        super().__init__()

        self.resblock1 = ResBlock(in_channels + skip_channels, out_channels, time_embd_dim = time_embed_dim, groups = n_groups)
        self.resblock2 = ResBlock(out_channels, out_channels, time_embd_dim = time_embed_dim, groups = n_groups)

    def forward(self, x_t: torch.Tensor, skip: torch.Tensor, t_e: torch.Tensor):
        """
        x_t: (B, C, H, W) [Decoder features]
        skip: (B, C, H, W) [Encoder features]
        t_e: (B, d)
        """

        # Ensure exact spatial alignment
        x_t = F.interpolate(
            x_t,
            size = skip.shape[-2:],
            mode="nearest"
        ) # make sure the spartial dim of 'x_t' and 'skip' tensor matches

        x = torch.cat([x_t, skip], dim = 1) # cat([(B, C1, H, W), (B, C2, H, W)]) --> (B, C1+c2, H, W)
        x = self.resblock1(x, t_e) # (B, C1+C2, H, W) ---> (B, C_out, H, W) | Injects Timestep
        x = self.resblock2(x, t_e) # (B, C_out, H, W) --> (B, C_out, H, W) | Injects Timestep
        return x # (B, C_out, H, W)

class BottleneckBlock(nn.Module):
    """
    the lowest point of the U-Net, where the Global information exchange happens, using AttentionBlock
    performs:
    ResBlock --> AttentionBlock --> ResBlock
    """
    def __init__(self, channels, time_embd_dim, n_group: int = 8):
        super().__init__()
        self.resblock1 = ResBlock(channels, channels, time_embd_dim, n_group)
        self.resblock2 = ResBlock(channels, channels, time_embd_dim, n_group)
        self.attention_block = AttentionBlock(channels, n_group)

    def forward(self, x_t: torch.Tensor, t_e: torch.Tensor):
        """
        x_t: (B, C, H, W)
        t_e: (B, d)

        returns: (B, C, H, W)
        """
        x_t = self.resblock1(x_t, t_e) # (B, C, H, W)
        x_t = self.attention_block(x_t) # (B, C, H, W)
        x_t = self.resblock1(x_t, t_e) # (B, C, H, W)
        return x_t # (B, C, H, W)

class UNet(NoiseModel):
    def __init__(self,
            in_channels: int = 3,
            out_channels: int = 3,
            base_channels: int = 64,
            time_embd_dim: int = 256,
            groups: int = 8
        ):
        super().__init__()

        assert base_channels % groups == 0
        assert (base_channels * 2) % groups == 0
        assert (base_channels * 4) % groups == 0
        assert (base_channels * 8) % groups == 0

        C1 = base_channels # 64
        C2 = base_channels * 2 # 128
        C3 = base_channels * 4 # 256
        C4 = base_channels * 8 # 512

        self.time_embedding = nn.Sequential(
            TimeStepEmbeddings(time_embd_dim),
            nn.Linear(time_embd_dim, time_embd_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embd_dim * 4, time_embd_dim)
        )

        self.initial_conv = nn.Conv2d(  # 3 -> 64
            in_channels,
            C1,
            kernel_size=3,
            padding=1
        )

        # --- Encoder --- 
        self.encoder1 = EncoderStage( # 64 --> 64
            C1, C1, time_embd_dim, groups
        )
        self.downsample1 = DownSampleBlock(C1, C2) # (64 -> 128)

        self.encoder2 = EncoderStage( # (128 -> 128)
            C2, C2, time_embd_dim, groups
        )
        self.downsample2 = DownSampleBlock(C2, C3)

        self.encoder3 = EncoderStage( # (128 -> 128)
            C3, C3, time_embd_dim, groups
        )
        self.downsample3 = DownSampleBlock(C3, C4)

        # --- Bottleneck Layer ---
        self.bottleneck = BottleneckBlock(
            C4, time_embd_dim, groups
        )

        # --- Decoder ---
        self.decoder3 = DecoderStage(
                in_channels = C4,
                skip_channels = C3,
                out_channels = C3,
                time_embed_dim = time_embd_dim,
                n_groups = groups
        ) 
        self.decoder2 = DecoderStage(
                in_channels = C3,
                skip_channels = C2,
                out_channels = C2,
                time_embed_dim = time_embd_dim,
                n_groups = groups
        ) 
        self.decoder1 = DecoderStage(
                in_channels = C2,
                skip_channels = C1,
                out_channels = C1,
                time_embed_dim = time_embd_dim,
                n_groups = groups
        )

        # output layer
        self.out_norm = nn.GroupNorm(num_channels = C1, num_groups = groups)
        self.output_conv = nn.Conv2d(
            C1, out_channels, kernel_size = 3, padding = 1
        )
        self.silu = nn.SiLU()

    def pad_to_multiple(self, x, multiple = 8):
        H, W = x.shape[-2:]

        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple

        # F.pad ordering: left, right, top, bottom
        x = F.pad(x, (0, pad_w, 0, pad_h))

        return x, (H, W)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor):
        """
        x_t: Noise Image, noise injected upto timestep 't'; shape = (B, C, H, W)
        t_e: TimeStep Embeddings: Shape = [B]

        returns: (B, C, H, W) 'predicted noise'
        """
        # Pad the input to the nearest multiple of '8'
        x_t, (H, W) = self.pad_to_multiple(x_t, multiple = 8)

        # ------------------ Encoder ---------------------------
        t_e = self.time_embedding(t) # (B, ) --> (B, d)
        x_t = self.initial_conv(x_t) # (B, 3, H1, W1) --> (B, 64, H2, W2)

        e1 = self.encoder1(x_t, t_e)
        x_t = self.downsample1(e1) # (B, 128, H3, W3)

        e2 = self.encoder2(x_t, t_e)
        x_t = self.downsample2(e2) # (B, 256, H4, W4)

        e3 = self.encoder3(x_t, t_e)
        x_t = self.downsample3(e3) # (B, 512, 32, 32)

        # ------------------ Bottleneck ---------------------------
        # print(f"(Bottleneck input) x_t.shape = {x_t.shape}")
        x_t = self.bottleneck(x_t, t_e) # (B, 512, H5, W5) --> (B, 512, H5, W5) [Mixes information]
        # print(f"(Bottleneck output) x_t.shape = {x_t.shape}")

        # ------------------ Decoder -----------------------------
        # print(f"x_t.shape = {x_t.shape} | e3.shape = {e3.shape} | t_e.shape = {t_e.shape}")
        x_t = self.decoder3(x_t, e3, t_e) # (B, 256, H4, W4)
        # print(f"x_t.shape = {x_t.shape} | e3.shape = {e3.shape} | t_e.shape = {t_e.shape}")
        x_t = self.decoder2(x_t, e2, t_e) # (B, 128, H3, W3)
        # print(f"x_t.shape = {x_t.shape} | e3.shape = {e3.shape} | t_e.shape = {t_e.shape}")
        x_t = self.decoder1(x_t, e1, t_e) # (B, 64, H2, W2)

        # Noise Prediction
        x_t = self.output_conv(self.silu(self.out_norm(x_t)))
        x_t = x_t[:, :, :H, :W] # Remove the padding and restore original resolution

        epsilion = x_t # (B, 64, H, W)
        return epsilion
    
class NoisePredictionPipeline(nn.Module):
    def __init__(self, config, noise_prediction_model: NoiseModel):
        super().__init__()
        self.T = config["T"]
        self.time_embed = config["time_embed"]
        self.noise_model = noise_prediction_model

    def forward(self, x_t: torch.Tensor, t: torch.Tensor):
        """
        x_t.shape = (B, C, H, W)
        t.shape = (B, )
        """
        epsilion = self.noise_model(x_t, t) # epsilion.shape = (B, C, H, W)
        return epsilion


if __name__ == "__main__":
    model = UNet(
        in_channels=3,
        out_channels=3,
        time_embd_dim=256,
        base_channels=64
    )

    x_t = torch.randn(2, 3, 256, 256)

    # Integer diffusion timesteps
    t = torch.randint(1, 1001, (2,))

    predicted_noise = model(x_t, t)

    print("Input:", x_t.shape)
    print("Predicted noise:", predicted_noise.shape)
    print("Parameters:", sum(p.numel() for p in model.parameters()))