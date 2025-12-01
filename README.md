<h1 align="center">🐶🐱 Dog vs. Cat Image Classification using CNN</h1> <p> This project implements a <b>Convolutional Neural Network (CNN)</b> to classify images as either <b>"dog"</b> or <b>"cat."</b> It includes a <b>Flask web application</b> for interactive classification, allowing users to upload their own images or get predictions on random images from the dataset. </p> <hr> <h1>⭐ Features</h1> <ul> <li><b>Image Classification:</b> Accurately classifies images into “dog” or “cat”.</li> <li><b>Web Interface:</b> Responsive Flask interface.</li> <li><b>Image Upload:</b> Upload your own images.</li> <li><b>Random Prediction:</b> Get predictions from test dataset.</li> <li><b>Confidence Score:</b> Shows model confidence.</li> <li><b>Modern UI:</b> Smooth, clean, user-friendly.</li> </ul> <hr> <h1>🛠 Technologies Used</h1>

Python

Flask

TensorFlow / Keras

MobileNetV2

NumPy

Pillow (PIL)

Gunicorn

HTML, CSS, JavaScript

<hr> <h1>📁 Project Structure</h1>
.
├── app.py                      # Flask web application for predictions
├── main.py                     # Script for model training and local testing
├── catvsdog_fixed.h5           # Pre-trained CNN model file
├── requirements.txt            # Python dependencies
├── static/                     # Static files (CSS)
│   └── style.css
├── templates/                  # HTML templates
│   └── index.html
├── uploads/                    # Temporary folder for uploaded images (ignored by Git)
└── dogs-vs-cats/               # Dataset (train and test images)
    ├── test/
    └── train/

<hr> <h1>🧪 Setup & Local Development</h1> <h2>1. Clone the repository</h2>
git clone https://github.com/Akshatsaxena279/Dog-vs-Cat-Image-Classification-using-CNN-.git
cd Dog-vs-Cat-Image-Classification-using-CNN-

<h2>2. Create a virtual environment</h2>
python -m venv venv


Windows:

.\venv\Scripts\activate


macOS/Linux:

source venv/bin/activate

<h2>3. Install dependencies</h2>
pip install -r requirements.txt

<h2>4. Run the Flask application</h2>
python app.py


Visit your browser at:

👉 http://127.0.0.1:5001/

<hr> <h1>🧠 Model Training (Optional)</h1>

The main.py script contains the CNN training logic.

To train the model:

python main.py


Requires the dogs-vs-cats dataset in the correct directory structure.

<hr> <h1>📞 Contact</h1>

Developed by: Akshat Saxena
📧 Email: akshatsaxena1977@gmail.com
