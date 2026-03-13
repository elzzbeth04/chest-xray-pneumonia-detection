import numpy as np
import tensorflow as tf
import cv2
import base64
import io

def get_img_array(img_path_or_bytes, size):
    """
    Helper function to load an image and convert it to a numpy array.
    Supports either a file path or image bytes.
    """
    if isinstance(img_path_or_bytes, bytes):
        img = tf.keras.preprocessing.image.load_img(
            io.BytesIO(img_path_or_bytes), target_size=size
        )
    else:
        img = tf.keras.preprocessing.image.load_img(img_path_or_bytes, target_size=size)
        
    array = tf.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 224, 224, 3)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_gradcam(img_array, heatmap, alpha=0.4):
    """
    Overlays the heatmap on the original image and returns a base64 string.
    """
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert img_array to uint8 0-255
    img = img_array[0]
    if np.max(img) <= 1.0:
        img = np.uint8(255 * img)
    else:
        img = np.uint8(img)
        
    # RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Resize jet to match image size
    jet = cv2.resize(jet, (img_bgr.shape[1], img_bgr.shape[0]))

    # Superimpose the heatmap on original image
    superimposed_img = jet * alpha + img_bgr
    
    # Normalize to 0-255
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    
    # Convert back to RGB
    superimposed_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    # Encode as JPEG
    is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(superimposed_rgb, cv2.COLOR_RGB2BGR))
    if not is_success:
        raise Exception("Could not encode image to JPEG")
        
    # Convert to base64
    base64_img = base64.b64encode(buffer).decode("utf-8")
    
    return f"data:image/jpeg;base64,{base64_img}"

def generate_gradcam_base64(img_array, model, last_conv_layer_name="out_relu"):
    """
    Full pipeline to generate Grad-CAM base64 string for a given preprocessed image array.
    """
    # MobileNetV2 last conv layer is typically 'out_relu'
    try:
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        base64_img = overlay_gradcam(img_array, heatmap)
        return base64_img
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        return None
