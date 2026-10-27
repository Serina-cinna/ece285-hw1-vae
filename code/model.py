"""Convolutional VAE for 64 or 128 pixel grayscale chest X-rays."""

import torch
from torch import nn


class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int = 128, image_size: int = 128):
        super().__init__()
        if image_size not in (64, 128):
            raise ValueError("image_size must be 64 or 128")
        levels = int(torch.log2(torch.tensor(image_size)).item()) - 2
        channels = [32 * (2 ** i) for i in range(levels)]
        encoder = []
        incoming = 1
        for i, outgoing in enumerate(channels):
            encoder.extend([nn.Conv2d(incoming, outgoing, 4, 2, 1)])
            if i:
                encoder.append(nn.BatchNorm2d(outgoing))
            encoder.append(nn.ReLU(inplace=True))
            incoming = outgoing
        self.encoder = nn.Sequential(*encoder, nn.Flatten())
        self.feature_size = channels[-1] * 4 * 4
        self.mean = nn.Linear(self.feature_size, latent_dim)
        self.log_variance = nn.Linear(self.feature_size, latent_dim)
        self.decode_input = nn.Linear(latent_dim, self.feature_size)

        decoder = [nn.Unflatten(1, (channels[-1], 4, 4))]
        for i in range(len(channels) - 1, 0, -1):
            decoder.extend([
                nn.ConvTranspose2d(channels[i], channels[i - 1], 4, 2, 1),
                nn.BatchNorm2d(channels[i - 1]),
                nn.ReLU(inplace=True),
            ])
        decoder.append(nn.ConvTranspose2d(channels[0], 1, 4, 2, 1))
        self.decoder = nn.Sequential(*decoder)
        self.image_size = image_size

    def encode(self, image):
        features = self.encoder(image)
        return self.mean(features), self.log_variance(features).clamp(-10, 10)

    @staticmethod
    def reparameterize(mean, log_variance):
        standard_deviation = torch.exp(0.5 * log_variance)
        return mean + standard_deviation * torch.randn_like(standard_deviation)

    def decode(self, latent):
        return self.decoder(self.decode_input(latent))

    def forward(self, image):
        mean, log_variance = self.encode(image)
        return self.decode(self.reparameterize(mean, log_variance)), mean, log_variance


def vae_loss(logits, image, mean, log_variance):
    """Return summed Bernoulli reconstruction and KL losses."""
    reconstruction = nn.functional.binary_cross_entropy_with_logits(logits, image, reduction="sum")
    kl = -0.5 * torch.sum(1 + log_variance - mean.square() - log_variance.exp())
    return reconstruction + kl, reconstruction, kl
