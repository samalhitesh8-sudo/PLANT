Plant Disease Detection
A machine learning application for detecting diseases in plants using computer vision and deep learning techniques. This project uses image classification to identify various plant diseases, helping farmers and agricultural professionals make informed decisions about crop health.
🌱 Features

Real-time Disease Detection: Upload plant images to get instant disease predictions
Multiple Disease Classification: Supports detection of various plant diseases
User-friendly Interface: Simple web application for easy interaction
High Accuracy: Trained deep learning model for reliable predictions
Preprocessing Pipeline: Automated image preprocessing for optimal results

📁 Project Structure
plant-disease-detection/
│
├── README.md                         # Project documentation
├── requirements.txt                  # Python dependencies
├── app.py                           # Main Flask/Streamlit application
├── model/
│   ├── saved_model.h5               # Trained model file
│   └── model_training.py            # Script to train the model
├── data/
│   ├── dataset/                     # Folder containing plant images
│   └── dataset_labels.csv           # Labels for dataset
├── utils/
│   ├── image_preprocessing.py       # Image preprocessing utilities
│   └── inference.py                 # Inference utilities
└── docs/
    └── usage.md                     # Detailed usage documentation
🚀 Getting Started
Prerequisites

Python 3.8 or higher
pip (Python package installer)
Virtual environment (recommended)

Installation

Clone the repository

bash   git clone https://github.com/your-username/plant-disease-detection.git
   cd plant-disease-detection

Create a virtual environment

bash   python -m venv plant-disease-env
   source plant-disease-env/bin/activate  # On Windows: plant-disease-env\Scripts\activate

Install dependencies

bash   pip install -r requirements.txt

Download/Prepare the dataset

Place your plant images in the data/dataset/ directory
Ensure dataset_labels.csv contains the correct labels for your images



💻 Usage
Running the Application

Start the web application

bash   python app.py

Open your browser and navigate to the displayed URL (typically http://localhost:5000 or http://localhost:8501)
Upload an image of a plant leaf and get instant disease prediction results

Training a New Model
If you want to train the model with your own dataset:
bashcd model/
python model_training.py
Using the Inference Module
For programmatic use:
pythonfrom utils.inference import predict_disease
from utils.image_preprocessing import preprocess_image

# Load and preprocess image
processed_image = preprocess_image('path/to/your/image.jpg')

# Make prediction
prediction = predict_disease(processed_image)
print(f"Predicted disease: {prediction}")
📊 Model Information

Architecture: Convolutional Neural Network (CNN)
Framework: TensorFlow/Keras
Input: RGB images (224x224 pixels recommended)
Output: Disease classification with confidence scores
Model File: model/saved_model.h5

🗂️ Dataset
The dataset should contain:

Images: Plant leaf images organized by disease category
Labels: CSV file with image names and corresponding disease labels
Format: Supported formats include JPG, PNG, JPEG

Dataset Structure Example:
data/dataset/
├── healthy/
│   ├── healthy_leaf_001.jpg
│   └── healthy_leaf_002.jpg
├── disease_1/
│   ├── diseased_leaf_001.jpg
│   └── diseased_leaf_002.jpg
└── disease_2/
    ├── diseased_leaf_003.jpg
    └── diseased_leaf_004.jpg
🔧 Configuration
Key configuration options can be modified in:

Model parameters: Edit model/model_training.py
Preprocessing settings: Modify utils/image_preprocessing.py
Application settings: Update app.py

📚 Dependencies
Main libraries used:

TensorFlow/Keras: Deep learning framework
OpenCV: Image processing
NumPy: Numerical computations
Pandas: Data manipulation
Flask/Streamlit: Web application framework
Pillow: Image handling

For the complete list, see requirements.txt.
🤝 Contributing

Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

Thanks to the agricultural research community for disease classification datasets
Open source computer vision libraries that made this project possible
Contributors and testers who helped improve the application

📞 Support
If you encounter any issues or have questions:

Check the documentation
Open an issue on GitHub
Contact the maintainers

🔄 Version History

v1.0.0: Initial release with basic disease detection
v1.1.0: Added web interface and improved accuracy
v1.2.0: Enhanced preprocessing and model optimization
