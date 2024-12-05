# CNN-Based Tuberculosis Classification System
This repository contains a CNN-based machine learning project for the classification of tuberculosis from medical images. The system leverages deep learning to assist in early detection, providing a user-friendly interface through a Flask web application.<br />
<p align="center">
  <img src="https://github.com/user-attachments/assets/800f31a9-c9ac-4e43-bab4-3ca1176bd598" alt=" Tuber culisis " />
</p>

## Overview
Tuberculosis (TB) is a significant global health challenge. Early and accurate detection of TB can save lives and reduce the spread of the disease. This project employs a Convolutional Neural Network (CNN) model trained on medical imaging datasets to classify whether an image indicates tuberculosis.<br />

The project is designed for educational and research purposes, showcasing the integration of a machine learning model into a web-based platform.

## Features
- **Pretrained CNN Model**: A pretrained model fine-tuned for tuberculosis classification.
- **Web Interface**: User-friendly web interface built using Flask.
- **Customizable**: Easily adaptable for other medical imaging tasks.
- **Lightweight**: Designed to run on local machines with minimal setup.
## Technologies Used
- **Python**: Core programming language.
- **Flask**: Web framework for building the application.
- **TensorFlow/Keras**: Deep learning library for model training and inference.
- **HTML, CSS, JavaScript**: Frontend technologies for the user interface.
- **Bootstrap** : For responsive UI design.
## Installation
### Prerequisites
Ensure you have the following installed:

- Python 3.8 or later
- pip (Python package installer)
## Steps
  1. Clone the repository :
  ```bash
  git clone https://github.com/Ravinderram/CNN-Based-Tuberculosis-Classification-System.git
  cd CNN-Based-Tuberculosis-Classification-System
  ```
  2. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
  3. Run the Flask application(Don't forgot to give the ```.H5 model``` path or you can create your own model by running ```Pretrained_model_code.ipynb``` ipynb file) :
  ``` bash
  python Flask_app.py
  ```
  4. Open your browser and navigate to:
  ```bash
  http://127.0.0.1:5000/
  ```
## File Structure 
- ```Flask_app.py```: Main application script.
- ```Pretrained_model_code.ipynb```: Jupyter notebook for training and testing the CNN model.
- ```model/```: Contains pretrained model files.
- ```static/```: Assets like CSS, JavaScript, and images for the web app.
- ```templates/```: HTML files for the web interface.
- ```images/```: Image samples or datasets.
## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests for improvements or bug fixes 👏 .

  
  
  
