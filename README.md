# 🧠 Sonar Rock vs Mine Classification using ANN

This project builds an Artificial Neural Network (ANN) to classify sonar signals as Rock (R) or Mine (M) using a dataset of 60 numerical features extracted from sonar frequency responses.

#  📌 Dataset Overview
Dataset: Sonar Dataset
Samples: 208
Features: 60 numeric features
Target:
R → Rock
M → Mine

Each row represents sonar signal readings bounced off a surface.

# ⚙️ Technologies Used
Python 🐍
Pandas
NumPy
Scikit-learn
TensorFlow / Keras
Matplotlib
Seaborn

# 📊 Data Preprocessing
Loaded dataset using pandas
Split into training and testing sets (80/20)
Applied Label Encoding:
R → 0
M → 1
Converted labels to numeric format for ANN training

# 🧠 Model Architecture
model = keras.Sequential([
    keras.layers.Dense(60, activation='relu', input_dim=60),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

# ⚙️ Compilation
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 🚀 Training
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=10,
    verbose=1
)

# 📈 Evaluation Results
Test Loss: ~0.31
Test Accuracy: ~85.7%
🔮 Prediction Workflow
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob >= 0.5).astype(int)

# 📊 Output Example
[[0]
 [1]
 [0]
 [1]
 ...
]
0 → Rock
1 → Mine

# 📌 Key Learnings
Handling binary classification with ANN
Importance of label encoding
Dropout layers to reduce overfitting
Sigmoid activation for binary output
Threshold-based prediction (0.5)

# 🚀 Future Improvements
Add Batch Normalization
Try CNN for feature extraction
Hyperparameter tuning (GridSearch / KerasTuner)
Deploy using Streamlit / Flask web app
