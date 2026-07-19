import streamlit as st
import pandas as pd
import pickle
import os
import io

from csv_utils import filter_nonempty_text_rows, preferred_text_column
from preprocessing import preprocess_text
from advanced import detect_emotion, detect_fake_review, translate_to_english
from visualization import plot_sentiment_distribution, plot_wordcloud, plot_model_comparison

# Try importing audio recorder, handle if not installed
try:
    from audio_recorder_streamlit import audio_recorder
    import speech_recognition as sr
    AUDIO_SUPPORT = True
except ImportError:
    AUDIO_SUPPORT = False

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Sentiment Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .metric-card {
        background-color: #262730;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .sentiment-positive { color: #00CC96; font-weight: bold; font-size: 24px;}
    .sentiment-negative { color: #EF553B; font-weight: bold; font-size: 24px;}
    .sentiment-neutral { color: #636EFA; font-weight: bold; font-size: 24px;}
    .fake-alert { color: #FF4B4B; font-weight: bold; }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Models ---
@st.cache_resource
def load_models():
    models_dir = 'models'
    try:
        with open(os.path.join(models_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
        
        with open(os.path.join(models_dir, 'logistic_regression_model.pkl'), 'rb') as f:
            lr_model = pickle.load(f)
            
        with open(os.path.join(models_dir, 'naive_bayes_model.pkl'), 'rb') as f:
            nb_model = pickle.load(f)
            
        with open(os.path.join(models_dir, 'svm_model.pkl'), 'rb') as f:
            svm_model = pickle.load(f)
            
        with open(os.path.join(models_dir, 'metrics.pkl'), 'rb') as f:
            metrics = pickle.load(f)
            
        return vectorizer, lr_model, nb_model, svm_model, metrics
    except Exception as e:
        st.error(f"Error loading models: {e}. Please run train.py first.")
        return None, None, None, None, None

vectorizer, lr_model, nb_model, svm_model, metrics = load_models()

# Mapping models for selection
model_map = {
    "Logistic Regression": lr_model,
    "Naive Bayes": nb_model,
    "SVM": svm_model
}

# --- Sidebar ---
st.sidebar.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
st.sidebar.title("AI Sentiment Pro")
st.sidebar.markdown("Production-Ready Sentiment Analysis")

app_mode = st.sidebar.selectbox(
    "Choose a mode",
    ["Single Prediction", "Batch Analysis (CSV)", "Model Analytics"]
)

selected_model_name = st.sidebar.selectbox("Select Model", list(model_map.keys()))
active_model = model_map[selected_model_name]

st.sidebar.markdown("---")
st.sidebar.info("Powered by Scikit-Learn, NLTK, and Streamlit.")

# --- Helper Functions ---
def predict_sentiment(text, model, vectorizer):
    clean_text = preprocess_text(text)
    vec_text = vectorizer.transform([clean_text])
    pred = model.predict(vec_text)[0]
    
    # Try to get probabilities
    try:
        probs = model.predict_proba(vec_text)[0]
        confidence = max(probs)
    except AttributeError:
        # Some models might not support predict_proba without specific config
        confidence = 1.0 
        probs = [0, 0, 0] # Dummy
        
    return pred, confidence, probs

# --- Main Logic ---
if app_mode == "Single Prediction":
    st.title("🧠 Real-Time Sentiment & Emotion Analysis")
    st.markdown("Enter text or use your voice to analyze sentiment, detect emotions, and flag fake reviews.")
    
    input_method = st.radio("Input Method", ["Text", "Voice"])
    
    user_input = ""
    
    if input_method == "Text":
        user_input = st.text_area("Enter your text here:", height=150, placeholder="I absolutely love this new feature! It's amazing.")
    elif input_method == "Voice":
        if AUDIO_SUPPORT:
            st.write("Click the microphone to record.")
            audio_bytes = audio_recorder()
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                with st.spinner("Transcribing audio..."):
                    try:
                        r = sr.Recognizer()
                        audio_file = io.BytesIO(audio_bytes)
                        with sr.AudioFile(audio_file) as source:
                            audio_data = r.record(source)
                            user_input = r.recognize_google(audio_data)
                            st.success(f"Transcribed Text: {user_input}")
                    except sr.UnknownValueError:
                        st.error("Could not understand audio.")
                    except sr.RequestError as e:
                        st.error(f"Could not request results; {e}")
        else:
            st.error("Audio support dependencies are missing. Please ensure audio-recorder-streamlit and SpeechRecognition are installed.")
            
    enable_translation = st.checkbox("Auto-Translate to English (Multilingual Support)")
    
    if st.button("Analyze", type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning("Please provide some input text.")
        else:
            with st.spinner("Analyzing..."):
                # Translation
                if enable_translation:
                    text_to_analyze = translate_to_english(user_input)
                    if text_to_analyze != user_input:
                        st.info(f"Translated Text: {text_to_analyze}")
                else:
                    text_to_analyze = user_input
                
                # Prediction
                sentiment, conf, probs = predict_sentiment(text_to_analyze, active_model, vectorizer)
                emotion = detect_emotion(text_to_analyze, sentiment)
                is_fake, fake_score = detect_fake_review(text_to_analyze)
                
                # Display Results
                st.markdown("### Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    css_class = f"sentiment-{sentiment.lower()}"
                    st.markdown(f"""
                    <div class="metric-card">
                        <p style="color:#888; margin-bottom:5px;">Sentiment</p>
                        <p class="{css_class}">{sentiment}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p style="color:#888; margin-bottom:5px;">Emotion</p>
                        <p style="font-size:24px; font-weight:bold;">{emotion}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col3:
                    fake_color = "#FF4B4B" if is_fake else "#00CC96"
                    fake_text = "Potential Fake" if is_fake else "Looks Authentic"
                    st.markdown(f"""
                    <div class="metric-card">
                        <p style="color:#888; margin-bottom:5px;">Review Authenticity</p>
                        <p style="color:{fake_color}; font-size:20px; font-weight:bold;">{fake_text}</p>
                        <p style="font-size:12px; color:#888;">Score: {fake_score:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("### Confidence")
                st.progress(float(conf))
                st.write(f"{conf*100:.1f}% Confident")

elif app_mode == "Batch Analysis (CSV)":
    st.title("📁 Batch Sentiment Analysis")
    st.markdown("Upload a CSV file containing a column of text to analyze in bulk.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("**Preview:**")
            st.dataframe(df.head())

            preferred_column = preferred_text_column(df.columns)
            default_index = 0
            if preferred_column is not None:
                default_index = list(df.columns).index(preferred_column)
            text_column = st.selectbox(
                "Select the text column to analyze:",
                df.columns,
                index=default_index,
            )

            if st.button("Run Batch Analysis"):
                with st.spinner("Processing..."):
                    df = filter_nonempty_text_rows(df, text_column)
                    if df.empty:
                        st.warning("No analyzable text rows found in the selected column.")
                        st.stop()

                    # Process
                    df['Cleaned_Text'] = df[text_column].astype(str).apply(preprocess_text)
                    vec_texts = vectorizer.transform(df['Cleaned_Text'])
                    df['Predicted_Sentiment'] = active_model.predict(vec_texts)
                    
                    st.success("Analysis Complete!")
                    
                    # Display results
                    st.markdown("### Visualizations")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_pie = plot_sentiment_distribution(df, 'Predicted_Sentiment')
                        st.plotly_chart(fig_pie, use_container_width=True)
                        
                    with col2:
                        st.markdown("#### Word Cloud (Overall)")
                        fig_wc = plot_wordcloud(df['Cleaned_Text'])
                        st.pyplot(fig_wc)
                    
                    # Download
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="sentiment_results.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(f"Error processing file: {e}")

elif app_mode == "Model Analytics":
    st.title("📊 Model Analytics & Comparison")
    
    if metrics:
        st.markdown("### Model Performance Comparison")
        fig_bar = plot_model_comparison(metrics)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("### Detailed Metrics")
        selected_metric_model = st.selectbox("Select model for details:", list(metrics.keys()))
        
        m = metrics[selected_metric_model]
        st.write(f"**Overall Accuracy:** {m['accuracy']:.4f}")
        
        st.markdown("#### Classification Report")
        # Convert classification report dict to dataframe
        cr_df = pd.DataFrame(m['report']).transpose()
        st.dataframe(cr_df.style.format("{:.3f}"))
        
        st.markdown("#### Confusion Matrix")
        cm = m['confusion_matrix']
        st.write(cm) # In a real scenario, plot this as a heatmap with Plotly
    else:
        st.warning("Metrics not found. Please train models first.")
