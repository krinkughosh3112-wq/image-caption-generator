import streamlit as st
import numpy as np
import pickle
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="wide"
)

# Custom CSS dark theme
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { background-color: #0e1117; }
    h1 { color: #00d4ff; text-align: center; }
    h2, h3 { color: #00d4ff; }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #0e1117);
        border: 1px solid #00d4ff;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px;
    }
    .caption-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 2px solid #00d4ff;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        font-size: 20px;
        color: #ffffff;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1> Image Caption Generator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#888;'>CNN + LSTM | VGG16 + LSTM | Flickr8K Dataset</p>", unsafe_allow_html=True)

# Load tokenizer
@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as f:
        return pickle.load(f)

vocab_size = 8485
max_length = 35

# Build model
def build_model():
    inputs1 = Input(shape=(4096,), name="image")
    fe1 = Dropout(0.4)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    inputs2 = Input(shape=(max_length,), name="text")
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.4)(se1)
    se3 = LSTM(256)(se2)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

@st.cache_resource
def load_models():
    vgg = VGG16()
    vgg = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)
    caption_model = build_model()
    caption_model.load_weights('model_epoch_21.h5')
    return vgg, caption_model

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length, padding='post')
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text

def extract_features(image, vgg_model):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    return vgg_model.predict(image, verbose=0)

# Sidebar - Model Info
with st.sidebar:
    st.markdown("## Model Info")
    st.markdown(f"**Architecture:** VGG16 + LSTM")
    st.markdown(f"**Dataset:** Flickr8K")
    st.markdown(f"**Total Images:** 8,091")
    st.markdown(f"**Vocab Size:** 8,485")
    st.markdown(f"**Max Caption Length:** 35")
    st.markdown(f"**Epochs Trained:** 21")
    st.markdown(f"**Batch Size:** 32")
    st.markdown(f"**Optimizer:** Adam")
    st.markdown(f"**Loss Function:** Categorical Crossentropy")

    st.markdown("---")
    st.markdown("##  Model Performance")
    st.metric("BLEU-1 Score", "0.5514", "↑ Better than baseline")
    st.metric("BLEU-2 Score", "0.3230")
    st.metric("METEOR Score", "0.3806")

# Loss Graph Section
st.markdown("---")
st.markdown("##  Training Loss Graph")

losses = [5.1362, 3.9263, 3.5110, 3.2472, 3.0568,
          2.9128, 2.8011, 2.7115, 2.6400, 2.5766,
          2.5200, 2.4648, 2.4137, 2.3633, 2.3189,
          2.2778, 2.2371, 2.2070, 2.1765, 2.1451,
          2.1195, 2.0922]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=list(range(1, len(losses)+1)),
    y=losses,
    mode='lines+markers',
    name='Training Loss',
    line=dict(color='#00d4ff', width=3),
    marker=dict(size=8, color='#ff6b6b')
))
fig.update_layout(
    title='Model Training Loss over Epochs',
    xaxis_title='Epochs',
    yaxis_title='Loss',
    plot_bgcolor='#1e1e2e',
    paper_bgcolor='#0e1117',
    font=dict(color='white'),
    height=400
)
st.plotly_chart(fig, use_container_width=True)

# Caption Generator Section
st.markdown("---")
st.markdown("##  Generate Caption")

uploaded_file = st.file_uploader(
    "Upload an image to generate caption",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        with st.spinner("AI is analyzing your image..."):
            tokenizer = load_tokenizer()
            vgg_model, caption_model = load_models()
            feature = extract_features(image, vgg_model)
            caption = predict_caption(caption_model, feature, tokenizer, max_length)
            caption = caption.replace('startseq', '').replace('endseq', '').strip()

        st.markdown("###  Generated Caption:")
        st.markdown(f"""
        <div class='caption-box'>
            💬 {caption}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("###  Evaluation Scores:")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("BLEU-1", "0.5514", "Excellent ✅")
        with c2:
            st.metric("BLEU-2", "0.3230", "Good ✅")
        with c3:
            st.metric("METEOR", "0.3806", "Good ✅")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:#888;'>Built with using VGG16 + LSTM | Flickr8K Dataset</p>", unsafe_allow_html=True)