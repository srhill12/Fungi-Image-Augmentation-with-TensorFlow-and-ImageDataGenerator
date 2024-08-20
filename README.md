
# Fungi Image Augmentation with TensorFlow and ImageDataGenerator

This project demonstrates how to preprocess and augment images using TensorFlow's `ImageDataGenerator`. Image augmentation is a crucial step in increasing the diversity of the training dataset without actually collecting new data. This technique is commonly used in training convolutional neural networks (CNNs) to improve the model's generalization capability.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Image Augmentation Process](#image-augmentation-process)
- [Visualization](#visualization)
- [Next Steps](#next-steps)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone repository_url
    ```

2. Install the required dependencies:
    ```bash
    pip install tensorflow matplotlib pillow requests numpy
    ```

## Usage

1. Import the required modules and load an image from a URL:
    ```python
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from matplotlib import pyplot as plt
    from PIL import Image
    import requests
    import numpy as np
    ```

2. Preprocess the image by resizing and normalizing:
    ```python
    img_url = "https://static.bc-edx.com/ai/ail-v-1-0/m19/lesson_2/datasets/H2_122c_4.jpg"
    example_image = Image.open(requests.get(img_url, stream=True).raw)
    example_image = example_image.resize((250, 250), Image.LANCZOS)
    float_image = np.array(example_image).astype(np.float32) / 255
    ```

3. Add a batch dimension to the image and apply augmentation:
    ```python
    reshaped_image_array = np.expand_dims(float_image, axis=0)
    datagen = ImageDataGenerator(rotation_range=20, fill_mode='nearest')
    augmented_image = next(datagen.flow(reshaped_image_array, batch_size=1))[0]
    ```

4. Visualize the augmented image alongside the original:
    ```python
    plt.imshow((augmented_image * 255).astype('uint8'))
    plt.show()
    
    plt.imshow((reshaped_image_array[0, :, :, :] * 255).astype('uint8'))
    plt.show()
    ```

## Image Augmentation Process

Image augmentation is performed using TensorFlow's `ImageDataGenerator`, which allows for real-time augmentation during training. In this project, the following augmentation techniques are applied:

- **Rotation:** The image is randomly rotated by up to 20 degrees.
- **Fill Mode:** The empty areas created by rotation are filled using the nearest pixels' value.

## Visualization

The project includes functionality to visualize the augmented images. This helps in understanding how augmentation affects the images and ensures that the augmented images are meaningful.

- **Original Image:**
    ![Original Image](image_link_here)
  
- **Augmented Image:**
    ![Augmented Image](image_link_here)

## Next Steps

This project can be extended in the following ways:

1. **Integration with CNN Training:**
   - The augmented images can be used to train a Convolutional Neural Network (CNN) model, improving its ability to generalize from limited data.
   - You can reuse the CNN model from a previous project (CNN Fungi Classification) and integrate this augmentation pipeline to create a more robust model.

2. **Experiment with Different Augmentations:**
   - Try other augmentation techniques such as zoom, width/height shift, shear, and horizontal flip. Compare the performance of the CNN with different augmentations.

3. **Dataset Expansion:**
   - Expand the dataset by applying multiple augmentations to each image and training the CNN on this expanded dataset.

4. **Transfer Learning:**
   - Consider applying transfer learning using a pre-trained model like VGG16 or ResNet50. Fine-tune the model with the augmented data to achieve better accuracy on the specific task.

## License

This project is licensed under the MIT License.
```
