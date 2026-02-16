🩺 Skin Cancer Detection Web App

🔗 Live Demo: https://skin-cancer-2.streamlit.app/

📁 GitHub Repository: https://github.com/shivangitiwari0411/skin-cancer

📌 Project Overview

The Skin Cancer Detection Web App is a deep learning–based application that classifies skin lesion images to predict whether they are potentially cancerous or non-cancerous.

The application is built using:

🧠 Deep Learning (TensorFlow/Keras)

🎨 Streamlit for interactive UI

🖼 Image preprocessing techniques

🐍 Python

This project demonstrates practical implementation of AI in healthcare for educational and research purposes.

🚀 Features

Upload skin lesion image

Real-time prediction using trained model

Clean and user-friendly interface

Deployable using Streamlit Cloud

Lightweight and easy to run locally

🛠 Tech Stack

Python

TensorFlow / Keras

Streamlit

NumPy

Pillow

⚙️ How It Works

User uploads a skin lesion image.

Image is preprocessed (resized, normalized).

Pre-trained CNN model analyzes the image.

Model outputs prediction (Cancerous / Non-Cancerous).

Result is displayed on the web interface.

📂 Project Structure
skin-cancer/
│
├── app.py                # Streamlit application
├── model.h5              # Trained deep learning model
├── requirements.txt      # Dependencies
├── packages.txt          # Deployment configuration
└── README.md             # Project documentation

💻 Run Locally
1️⃣ Clone the repository
git clone https://github.com/shivangitiwari0411/skin-cancer.git
cd skin-cancer

2️⃣ Create virtual environment (optional but recommended)
python -m venv venv


Activate environment:

Windows:

venv\Scripts\activate


Mac/Linux:

source venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the application
streamlit run app.py

⚠️ Disclaimer

This application is developed for educational and research purposes only.
It is not a substitute for professional medical diagnosis.
Always consult a certified medical professional for health-related concerns.

📈 Future Improvements

Improve model accuracy with larger dataset

Add probability confidence scores

Add multiple skin disease classification

Add patient history input

Improve UI/UX
