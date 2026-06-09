import streamlit as st
import tensorflow as tf
import numpy as np

# 1. Setup the page design
st.set_page_config(page_title="NeuroSentiment AI", page_icon="🧠")
st.title("🧠 NeuroSentiment AI")
st.write("Enter a sentence below and my Neural Network will tell you if it's Positive or Negative.")

# 2. Get input from the user
user_input = st.text_area("Type your sentence here:", "I really love this amazing project!")

# 3. Create a Predict button
if st.button("Analyze Sentiment"):
    with st.spinner('Neural Network is thinking...'):
        # Load the brain we trained earlier
        model = tf.keras.models.load_model('sentiment_brain.keras')
        
        # Make the prediction
        # We wrap it in tf.constant to prevent the same bug we fixed earlier
        prediction = float(model.predict(tf.constant([user_input]), verbose=0)[0][0])
        
        # Calculate percentage
        positive_percent = prediction * 100
        negative_percent = 100 - positive_percent
        
    # 4. Show the results beautifully
    st.subheader("Results:")
    
    if positive_percent > 50:
        st.success(f"✅ POSITIVE ({positive_percent:.1f}% confidence)")
    else:
        st.error(f"❌ NEGATIVE ({negative_percent:.1f}% confidence)")
        
    # Draw a progress bar
    st.progress(prediction)
    st.caption("Progress bar shows Positive confidence (0% = Negative, 100% = Positive).")
