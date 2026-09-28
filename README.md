# CelebA Convolutional Autoencoder

A small convolutional autoencoder built with TensorFlow/Keras that learns to compress and reconstruct grayscale face images from the [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

The whole project lives in one notebook: `CelebA.ipynb`.

## What it does

1. **Loads images**: reads the first 5,000 files from the `img_align_celeba` folder, converts each to grayscale, and resizes it to 28×28 pixels.
2. **Preprocesses**: scales pixel values to the range [0, 1], splits the data 80% train / 20% test (`random_state=42`), and adds a channel dimension so the shape is `(N, 28, 28, 1)`.
3. **Builds the model**: an encoder and a decoder joined into one autoencoder.
4. **Trains**: the model learns to reproduce its own input (input and target are the same images).
5. **Evaluates**: reconstructs 5 test images, prints the min, max and mean of the output, and plots originals above reconstructions.

## Model architecture

| Part | Layers |
|------|--------|
| **Encoder** | Conv2D(32, 3×3, ReLU) → MaxPool(2×2) → Conv2D(64, 3×3, ReLU) → MaxPool(2×2) |
| **Decoder** | Conv2DTranspose(64, stride 2, ReLU) → Conv2DTranspose(32, stride 2, ReLU) → Conv2D(1, 3×3, sigmoid) |

The encoder compresses a 28×28×1 image down to a 7×7×64 representation, and the decoder upsamples it back to 28×28×1.

**Training settings:** Adam optimizer, binary cross-entropy loss, 3 epochs, batch size 128, with the test set used as validation data.

## Requirements

- Python 3.8+
- `tensorflow`
- `numpy`
- `matplotlib`
- `pillow`
- `scikit-learn`

```bash
pip install tensorflow numpy matplotlib pillow scikit-learn
```

## Setup and usage

1. Download the **aligned and cropped** CelebA images (`img_align_celeba`) from the official site or Kaggle.
2. Update the path in the second code cell to where you extracted the images:
   ```python
   image_folder = r"C:\img_align_celeba"
   ```
3. Open the notebook and run all cells in order:
   ```bash
   jupyter notebook CelebA.ipynb
   ```

## Notes and possible improvements

- **Duplicate cell:** the cell that defines `autoencoder = models.Sequential([encoder, decoder])` appears twice. The second one is redundant and can be deleted.
- **Small scale:** 5,000 images at 28×28 grayscale and only 3 epochs is a quick demo, so reconstructions will be blurry. More epochs, more images, or a higher resolution should improve quality.
- **File filtering:** the `[:5000]` slice is applied before the `.jpg` check, so if the folder contains non-JPG files you may load slightly fewer than 5,000 images.
- **Hard-coded path:** the Windows path is specific to one machine, so change it for your own setup.
- **Ideas to extend:** add a dense bottleneck for a compact latent vector, try a denoising autoencoder, or plot the training and validation loss from `history`.

## Dataset credit

CelebA: Liu, Z., Luo, P., Wang, X., Tang, X. "Deep Learning Face Attributes in the Wild," ICCV 2015.
