##########################################################################
# Description: Main logic for CLI-based Neural Style Transfer
# Notes:
##########################################################################
from fire import Fire

import utils
from neuralStyleTransfer import NeuralStyleTransfer, NSTConfig

def exec(content_path: str, style_path: str, output_path: str, **kwargs) -> None:
    # Initialize the Neural Style Transfer model
    nst = NeuralStyleTransfer()
    config = NSTConfig().update_config(**kwargs)
    
    # Load the content and style images
    content = utils.load_image(content_path)
    style = utils.load_image(style_path)
    
    # Perform Neural Style Transfer and save image
    stylized = nst.transfer_style(content, style, config)
    utils.save_image(stylized, output_path)

if __name__ == '__main__':
    Fire(exec)