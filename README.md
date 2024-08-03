# StylizeIt
An implementation of Neural Style Transfer with PyTorch. It allows users to use command line arguments to stylize their own images easily and optionally set hyperparameters different from the default values.
<table>
  <tr>
    <td><b>Content Image</b></td>
    <td><b>Style Image</b></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/c5d29537-d05e-49db-824f-e5e592e2b33a" alt="Content Image" width="400"/></td>
    <td><img src="https://github.com/user-attachments/assets/aa4880b7-c421-4b38-84e3-9b23c3f904c1" alt="Style Image" width="400"/></td>
  </tr>
</table>
<p align="center">
  <b>Result</b><br>
  <img src="https://github.com/user-attachments/assets/dd978e92-1dd4-4d65-8688-774bb1bcc76f" alt="Result Image" width="600"/>
</p>

## Quick Start and requirements
1. Clone (or download) this repository:
```txt
git clone https://github.com/Germinari1/StylizeIt.git
```
2. Install the requirements for this project with:
```txt
pip install -r requirements.txt
```
There are some other important components for this project, which also need to be installed:
- Python 3
- Pip
- CUDA is _recommended_ for GPU acceleration
Now you`re ready to go!

## How to use StylizeIt
Command-line arguments are used to control the program and stylize your images. Here's the simplest way to generate a stylized image:
```txt
python exec.py <path to content image> <path to style image> <path for output>
```
For example:
```txt
python exec.py "images/content_imgs/figures.jpg" "images/style_imgs/wave_crop.jpg" "images/stylized_imgs/output1.jpg"
```
Besides creating images, it is possible to change the hyperparameters of the model by using the corresponding flags. For instance:
```txt
python exec.py "images/content_imgs/figures.jpg" "images/style_imgs/wave_crop.jpg" "images/stylized_imgs/output1.jpg" --num_iterations 1000 --learning_rate 0.01
```
Here's the complete list of parameters you can manipulate from the command line:
- `content_weight`
- `style_weight`
- `total_variation_weight`
- `learning_rate`
- `num_iterations`
- `content_layer_weights`
- `style_layer_weights`
- `use_random_init`
- `max_image_size`
- `checkpoint_interval`
