##########################################################################
# Description: Implements neural style transfer logic and hyperparameter configuration
# Notes:
#    - commnad line arguments handled in 'NSTConfig'
##########################################################################
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from tqdm import tqdm
import utils
from vgg_network import Vgg19  

@dataclass
class NSTConfig:
    """Configuration for Neural Style Transfer."""
    content_weight: float = 1.0
    style_weight: float = 100.0
    total_variation_weight: float = 10.0
    learning_rate: float = 0.1
    num_iterations: int = 500
    content_layer_weights: Tuple[float] = (0.0, 0.0, 0.0, 1.0, 0.0)
    style_layer_weights: Tuple[float] = (0.2, 0.2, 0.2, 0.2, 0.2)
    use_random_init: bool = False
    max_image_size: int = 512
    checkpoint_interval: int = 50

    # Update configuration parameters based on command lines arguments that match parameter names
    def update_config(self, **kwargs) -> 'NSTConfig':
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f'Invalid configuration parameter: "{key}"')
        return self


class NeuralStyleTransfer:
    """Neural Style Transfer implementation."""

    def __init__(self, use_cuda: bool = True):
        self.device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
        self.vgg_model = Vgg19(use_avg_pooling=True).to(self.device)
        self.optimizer = None

    def transfer_style(
        self,
        content_image: np.ndarray,
        style_image: np.ndarray,
        config: Optional[NSTConfig] = None,
    ) -> np.ndarray:
        """Execute Neural Style Transfer algorithm."""
        config = config or NSTConfig()
        print(f"Running with config: {config}")

        #preprocess and define image (different wheter random init or not)
        content_tensor = self.prepare_image(content_image, config.max_image_size)
        style_tensor = self.prepare_image(style_image, config.max_image_size)

        if config.use_random_init:
            output_tensor = torch.rand_like(content_tensor, requires_grad=True)
        else:
            output_tensor = content_tensor.clone().requires_grad_(True)

        with torch.no_grad():
            content_features = self.vgg_model(content_tensor)
            style_features = self.vgg_model(style_tensor)

        # Define optimizer
        self.optimizer = torch.optim.Adam([output_tensor], lr=config.learning_rate)

        # Run optimization and take care of progress bar
        progress_bar = tqdm(range(1, config.num_iterations + 1))
        # Optimization loop
        for iteration in progress_bar:
            loss = self.optimization_step(content_features, style_features, output_tensor, config)
            avg_gradient = output_tensor.grad.abs().mean().item()
            progress_bar.set_description(
                f'Loss: {loss:.2f}, Avg Gradient: {avg_gradient:.7f}'
            )

            # Save checkpoints (intermediary images)
            if iteration % config.checkpoint_interval == 0:
                checkpoint_image = self.finalize_image(output_tensor)
                utils.save_image(checkpoint_image, f'images/checkpoints/checkpoints_{iteration}.jpg')

        # Post-process image
        final_image = self.finalize_image(output_tensor)
        return final_image

    def prepare_image(self, image: np.ndarray, max_size: int) -> Tensor:
        """Preprocess image for Neural Style Transfer. Scale image if it is bigger than max size allowed"""
        height, width = image.shape[:2]
        if max(height, width) > max_size:
            scale_factor = max_size / max(height, width)
            new_size = (int(width * scale_factor), int(height * scale_factor))
            image = utils.resize_img(image, new_size)

        image_tensor = torch.tensor(image, device=self.device)
        return image_tensor.unsqueeze(0).permute(0, 3, 1, 2)

    @staticmethod
    def finalize_image(image_tensor: Tensor) -> np.ndarray:
        return image_tensor.permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()

    def optimization_step(
        self,
        content_features: List[Tensor],
        style_features: List[Tensor],
        output_tensor: Tensor,
        config: NSTConfig,
    ) -> float:
        """Perform a single optimization step."""
        output_features = self.vgg_model(output_tensor)
        
        content_loss = self.compute_content_loss(
            output_features, content_features, config.content_layer_weights
        )
        style_loss = self.compute_style_loss(
            output_features, style_features, config.style_layer_weights
        )
        tv_loss = self.compute_total_variation(output_tensor)
        
        total_loss = (
            content_loss * config.content_weight +
            style_loss * config.style_weight +
            tv_loss * config.total_variation_weight
        )
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        with torch.no_grad():
            output_tensor.clamp_(0.0, 1.0)
        
        return total_loss.item()

    @staticmethod
    def compute_content_loss(
        input_features: List[Tensor],
        target_features: List[Tensor],
        layer_weights: Tuple[float]
    ) -> Tensor:
        """Compute loss with respect to the content."""
        assert len(input_features) == len(target_features) == len(layer_weights)
        device = input_features[0].device
        total_loss = torch.zeros(1, dtype=torch.float32, device=device)

        for input_feat, target_feat, weight in zip(input_features, target_features, layer_weights):
            if weight > 0:
                layer_loss = F.mse_loss(input_feat, target_feat)
                total_loss += layer_loss * weight

        return total_loss

    @staticmethod
    def compute_style_loss(
        input_features: List[Tensor],
        target_features: List[Tensor],
        layer_weights: Tuple[float]
    ) -> Tensor:
        """Compute loss with respect to the style."""
        assert len(input_features) == len(target_features) == len(layer_weights)
        device = input_features[0].device
        total_loss = torch.zeros(1, dtype=torch.float32, device=device)

        for input_feat, target_feat, weight in zip(input_features, target_features, layer_weights):
            if weight > 0:
                input_gram = NeuralStyleTransfer.gram_matrix(input_feat)
                target_gram = NeuralStyleTransfer.gram_matrix(target_feat)
                layer_loss = F.mse_loss(input_gram, target_gram)
                total_loss += layer_loss * weight

        return total_loss

    @staticmethod
    def compute_total_variation(image: Tensor) -> Tensor:
        horizontal_diff = (image[:, :, :, :-1] - image[:, :, :, 1:]).abs().mean()
        vertical_diff = (image[:, :, :-1, :] - image[:, :, 1:, :]).abs().mean()
        return horizontal_diff + vertical_diff

    @staticmethod
    def gram_matrix(features: Tensor) -> Tensor:
        """Compute the Gram matrix of a set of features."""
        batch, channels, height, width = features.size()
        flattened_features = features.view(batch, channels, height * width)
        transposed_features = flattened_features.transpose(1, 2)
        gram = torch.bmm(flattened_features, transposed_features)
        return gram / (height * width)