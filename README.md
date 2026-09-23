# 🍽️ AI-Powered Food Image Classification and Nutrition Dashboard

An intelligent web-based dashboard that recognizes food from images and generates instant ingredient lists and nutrition facts using deep learning and large language models.

---

## 🌟 Features

- 🔍 **Food Image Recognition**: Upload a food photo and let the AI classify it from 101 food categories.
- 🧬 **Deep Learning Model**: Powered by a custom-trained **VGG16 CNN** architecture.
- 🧠 **Smart Ingredient & Nutrition Generation**: Integrates **Groq LLM API (LLaMA 3.3 70B)** to provide human-readable food insights.
- 🎨 **Modern UI**: Built with responsive HTML + TailwindCSS frontend.
- 💬 **Interactive Output**: Displays predictions, ingredients, and nutritional content side-by-side.

---

## 🚀 How It Works

The system operates via a clear pipeline that takes an uploaded food image, identifies the dish using a custom computer vision model, and fetches structured details using an LLM API:

[User Image] ──► [Flask UI / Backend] ──► [VGG16 Model] ──► [Predicted Food Label] ──► [Groq LLaMA 3.3] ──► [UI Dashboard]


### 1. Image Upload & Preprocessing
- **User Action:** The user selects and submits a food photo using the modern TailwindCSS web dashboard (`index.html`).
- **Image Standardisation:** Upon reaching the Flask backend (`app.py`), the image is resized to **224×224 pixels** and normalized to match the required input shape for the VGG16 network.

### 2. Deep Learning Classification
- **Model Execution:** The preprocessed image tensor is passed into the pre-trained `vgg16_food101_trained.h5` model inside `model.py`.
- **Prediction:** The convolutional neural network extracts visual features (textures, shapes, colors) and evaluates them through a Softmax output layer containing **101 distinct food classes**.
- **Result Output:** The class with the highest probability score (e.g., `"sushi"`, `"pad_thai"`, or `"apple_pie"`) is selected as the primary prediction label.

### 3. LLM Integration & Nutrition Generation
- **Prompt Construction:** The Flask backend constructs a targeted prompt containing the predicted food label.
- **Groq API Request:** The prompt is sent to **Groq's LLaMA 3.3 (70B parameter) LLM** via API.
- **Data Structuring:** The LLM processes the query and generates:
  - **Ingredients:** A list of key components typically present in the identified dish.
  - **Nutritional Profile:** Standard nutritional estimates (Calories, Macronutrients: Proteins, Fats, Carbohydrates, and key Micronutrients).

### 4. Real-Time Rendering
- **Data Delivery:** The backend formats the predicted label, ingredients, and nutrition facts into JSON and sends them back to the frontend.
- **Interactive UI Update:** The dashboard dynamically updates the UI without requiring a full page refresh, displaying the classified photo, predicted dish name, ingredient list, and macro breakdown side-by-side.
---

## 🧠 Model Details

- Based on **VGG16** architecture (no top layers).
- Trained on [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) with:
  - 📸 Image augmentations
  - 🔢 101 output classes (softmax layer)
  - 📊 ~80 epochs, batch size 16
- Accuracy: Evaluated with custom visualization + human loop.

---

## 📁 Folder Structure
.
├── app.py # Flask backend for image upload and API integration
├── model.py # Model architecture and training code
├── vgg16_food101_trained.h5 # Trained model weights
├── templates/
│ └── index.html # Dashboard UI
├── static/ # (Optional) for styles/images if extended
└── api.txt # API key and class labels
