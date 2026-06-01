import tensorflow as tf

print("🧠 Starting AI Brain Training...")

# 1. The Training Data (Sentences and their labels: 1 = Positive, 0 = Negative)
raw_sentences = [
    "I absolutely loved this movie", "This is the best day of my life",
    "The food was terrible and cold", "I hate waiting in long lines",
    "Amazing service and friendly staff", "Worst experience ever, do not go",
    "I am so happy right now", "This is boring and a waste of time",
    "Highly recommended, fantastic quality", "Disgusting and rude employees"
]
raw_labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]

# FIX: Convert to TensorFlow constants to bypass the Keras 3 string bug
sentences = tf.constant(raw_sentences)
labels = tf.constant(raw_labels, dtype=tf.float32)


# 2. Prepare the Text for the AI (Text Vectorization)
vectorizer = tf.keras.layers.TextVectorization(max_tokens=100, output_sequence_length=5)
vectorizer.adapt(sentences)

# 3. Build the Neural Network
model = tf.keras.Sequential([
    vectorizer,
    tf.keras.layers.Embedding(input_dim=100, output_dim=8),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid') # Sigmoid gives us a percentage (0% to 100%)
])

# 4. Compile and Train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("Training in progress (this takes about 5 seconds)...")
model.fit(sentences, labels, epochs=50, verbose=0) # verbose=0 hides the messy math output

# 5. Save the Brain so our website can use it later
model.save('sentiment_brain.keras')
print("✅ Success! AI Brain saved as 'sentiment_brain.keras'")

# 6. Quick Test
test_sentence = tf.constant(["This product is incredibly good!"])
prediction = model.predict(test_sentence, verbose=0)[0][0]
print(f"Test Sentence: 'This product is incredibly good!'")
print(f"AI Confidence it is POSITIVE: {prediction * 100:.1f}%")