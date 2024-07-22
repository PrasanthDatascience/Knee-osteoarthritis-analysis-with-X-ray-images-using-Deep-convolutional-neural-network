import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Function to generate Grad-CAM heatmap
def make_gradcam_heatmap(grad_model, img_array, pred_index=None):
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Function to save and display Grad-CAM overlay
def save_and_display_gradcam(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img

# Load model
model = tf.keras.models.load_model("./src/models/model_Xception_ft.hdf5")
target_size = (224, 224)

# Grad-CAM setup
grad_model = tf.keras.models.clone_model(model)
grad_model.set_weights(model.get_weights())
grad_model.layers[-1].activation = None
grad_model = tf.keras.models.Model(
    inputs=[grad_model.inputs],
    outputs=[
        grad_model.get_layer("global_average_pooling2d_1").input,
        grad_model.output,
    ],
)

# Sidebar
st.sidebar.title("About Knee Arthrosis")
st.sidebar.write(
    "Knee arthrosis, also known as knee osteoarthritis (OA), is a degenerative joint disease that commonly affects the knee joint. It involves the breakdown of cartilage and the underlying bone, leading to pain, stiffness, and reduced range of motion. Risk factors include age, obesity, previous joint injury, and genetics. Early detection and management can help alleviate symptoms and improve quality of life."
)

# Body
st.title("Severity Analysis of Arthrosis in the Knee")
uploaded_file = st.file_uploader("Upload knee X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.subheader("Input Image")
    st.image(uploaded_file, use_column_width=True)

    img = Image.open(uploaded_file)
    img = img.resize(target_size)
    img_array = np.expand_dims(np.array(img), axis=0)
    img_array = tf.keras.applications.xception.preprocess_input(img_array)

    if st.button("Predict Arthrosis"):
        with st.spinner("Analyzing..."):
            y_pred = model.predict(img_array)
        y_pred = 100 * y_pred[0]

        st.subheader("Prediction")
        class_names = ["Healthy", "Doubtful", "Minimal", "Moderate", "Severe"]
        max_index = np.argmax(y_pred)
        st.write(
            f"The knee shows {class_names[max_index]} arthrosis with a probability of {y_pred[max_index]:.2f}%."
        )

        st.subheader("Explainability")
        heatmap = make_gradcam_heatmap(grad_model, img_array)
        transparency = st.slider("Adjust Heatmap Transparency", 0.0, 1.0, 0.4, 0.1)
        overlay_img = save_and_display_gradcam(np.array(img), heatmap, alpha=transparency)
        st.image(overlay_img, caption="Grad-CAM Heatmap Overlay", use_column_width=True)

        st.subheader("Analysis")
        fig, ax = plt.subplots()
        ax.barh(class_names, y_pred)
        ax.set_xlabel("Probability (%)")
        ax.set_title("Arthrosis Severity Analysis")
        st.pyplot(fig)
