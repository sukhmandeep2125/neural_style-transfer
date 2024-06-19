import tensorflow as tf
from tensorflow.keras.preprocessing import image as kp_image
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

# Function to load and preprocess the image
def load_and_process_img(path_to_img):
    img = Image.open(path_to_img)
    max_dim = 512
    long = max(img.size)
    scale = max_dim / long
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.LANCZOS)
    img = kp_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19_preprocess_input(img)  # Correct usage of preprocess_input
    return img

# Function to deprocess the image
def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Function to load and display the image
def load_img(path_to_img):
    max_dim = 512
    img = Image.open(path_to_img)
    long = max(img.size)
    scale = max_dim / long
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.LANCZOS)
    img = kp_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

# Function to get the VGG19 model
def get_model():
    vgg = VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    outputs = [vgg.get_layer(name).output for name in (content_layers + style_layers)]
    model = Model([vgg.input], outputs)
    return model, content_layers, style_layers

# Function to compute content loss
def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

# Function to compute gram matrix
def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

# Function to compute style loss
def get_style_loss(base_style, gram_target):
    gram_style = gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_style - gram_target))

# Function to compute total loss
def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights
    model_outputs = model(init_image)
    style_output_features = model_outputs[len(content_features):]
    content_output_features = model_outputs[:len(content_features)]

    style_score = 0
    content_score = 0

    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += get_style_loss(comb_style[0], target_style)
    
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += get_content_loss(comb_content[0], target_content)
    
    style_score *= style_weight
    content_score *= content_weight

    loss = style_score + content_score
    return loss, style_score, content_score

# Function to compute gradients
def compute_gradients(cfg):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_image']), all_loss

# Function to run style transfer
def run_style_transfer(content_path, style_path, num_iterations=1000, content_weight=1e3, style_weight=1e-2):
    model, content_layers, style_layers = get_model()
    for layer in model.layers:
        layer.trainable = False

    content_image = load_and_process_img(content_path)
    style_image = load_and_process_img(style_path)

    content_features = model(content_image)[:len(content_layers)]
    style_features = model(style_image)[len(content_layers):]

    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    init_image = tf.Variable(content_image, dtype=tf.float32)
    opt = tf.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)

    loss_weights = (style_weight, content_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }

    num_rows = 2
    num_cols = 5
    display_interval = num_iterations // (num_rows * num_cols)
    global_start = time.time()
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    best_loss, best_img = float('inf'), None

    imgs = []
    for i in range(num_iterations):
        grads, all_loss = compute_gradients(cfg)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)

        if loss < best_loss:
            best_loss = loss
            best_img = deprocess_img(init_image.numpy())

        if i % display_interval == 0:
            start_time = time.time()
            plot_img = init_image.numpy()
            plot_img = deprocess_img(plot_img)
            imgs.append(plot_img)
            print('Iteration: {}'.format(i))
            print('Total loss: {:.4e}, style loss: {:.4e}, content loss: {:.4e}, time: {:.4f}s'.format(
                loss, style_score, content_score, time.time() - start_time))

    print('Total time: {:.4f}s'.format(time.time() - global_start))
    return best_img

# Paths to content and style images
content_path = '/content/drive/MyDrive/project/dog.jpeg'  # Replace with your content image path
style_path = '/content/drive/MyDrive/project/style.jpg'   # Replace with your style image path

# Run style transfer
output_image = run_style_transfer(content_path, style_path)

# Display the final result
plt.figure(figsize=(10, 10))
plt.imshow(output_image)
plt.title('Output Image')
plt.axis('off')
plt.show()
