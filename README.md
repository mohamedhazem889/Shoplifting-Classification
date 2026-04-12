# Shoplifting Classification

A deep learning project for detecting and classifying shoplifting behavior in surveillance videos using the **R3D-18 (ResNet-3D-18)** pre-trained model.

## 📋 Project Overview

This repository contains an implementation of a shoplifting detection system that leverages video analysis and deep learning techniques. The deployment uses the **R3D-18 pre-trained model**, a powerful 3D convolutional architecture optimized for video classification tasks.

## 🎬 Dataset

The project utilizes the **Shoplifting Videos Dataset** from Kaggle, which contains:
- Curated surveillance video sequences
- Labeled instances of shoplifting and non-shoplifting behavior
- Real-world video footage for practical model training

**Dataset Source:** [Shoplifting Videos Dataset on Kaggle](https://www.kaggle.com/datasets/omarelg/shoplifting-videos-dataset)

### Dataset Characteristics:
- **Video Format:** Surveillance camera feeds
- **Labels:** Binary classification (shoplifting vs. normal behavior)
- **Applications:** Security systems, retail loss prevention, anomaly detection

## 🛠️ Technical Stack

- **Language:** Python (Jupyter Notebooks - 99.6%)
- **Deep Learning Framework:** TensorFlow/PyTorch
- **Model Architecture:** R3D-18 (ResNet-3D-18)
- **Pre-trained Weights:** Kinetics Dataset
- **Containerization:** Docker & Docker Compose
- **Framework:** Django (Web API)

## 📁 Repository Structure

```
├── Model_3dCNN.ipynb          # Reference 3D CNN implementation (archived)
├── pretrained.ipynb           # R3D-18 pre-trained model inference
├── shoplifting_api/           # Django API for model serving
├── detector/                  # Detection utilities and helper modules
├── templates/                 # HTML templates for web interface
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose orchestration
├── manage.py                  # Django management script
└── .gitignore                 # Git ignore rules
```

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- pip or conda
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mohamedhazem889/Shoplifting-Classification.git
   cd Shoplifting-Classification
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   - Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/omarelg/shoplifting-videos-dataset)
   - Download and extract the dataset to your project directory

### Running with Docker

```bash
docker-compose up --build
```

### Running Jupyter Notebooks

```bash
jupyter notebook pretrained.ipynb
```

## 📊 R3D-18 Model Architecture

**R3D-18** (ResNet-3D-18) is a specialized 3D convolutional architecture designed specifically for video classification. Here's how it works:

### Key Features:

1. **3D Convolutions**
   - Applies convolutions across spatial (height × width) and temporal (frames) dimensions
   - Captures both object features and motion patterns in video sequences
   - More effective than 2D CNNs for video data due to temporal awareness

2. **Residual Learning**
   - Implements residual connections (skip connections) to allow deeper networks
   - Mitigates the vanishing gradient problem during training
   - Enables training of 18-layer deep architecture efficiently

3. **Pre-trained on Kinetics**
   - R3D-18 is pre-trained on the Kinetics-400 dataset (400 human action classes)
   - Transfer learning from Kinetics improves shoplifting detection performance
   - Reduces training time and data requirements

4. **Temporal-Spatial Feature Extraction**
   - **Spatial Features:** Objects, people, and items within frames
   - **Temporal Features:** Motion patterns and behavioral sequences
   - Combined representation for robust shoplifting classification

5. **Efficient Architecture**
   - 18 layers balanced between accuracy and computational efficiency
   - Suitable for real-time inference on edge devices
   - Optimized for video processing at practical frame rates

### Model Advantages Over Generic 3D CNN:

- Pre-trained weights provide better initial feature representations
- Designed specifically for action recognition in videos
- Proven performance on diverse video datasets
- More efficient than training 3D CNN from scratch
- Better generalization to shoplifting scenarios

## 🔧 API Usage

The project includes a Django-based REST API for model inference using the pre-trained R3D-18 model:

```bash
python manage.py runserver
```

### API Endpoints:
- **Video Upload:** Process new surveillance video
- **Real-time Detection:** Get shoplifting predictions
- **Confidence Scores:** Obtain model confidence in predictions

## 📈 Model Deployment

The `pretrained.ipynb` notebook demonstrates:
- Loading pre-trained R3D-18 weights
- Video preprocessing and frame extraction
- Real-time inference on new videos
- Confidence thresholding for alerts

## 🎯 Use Cases

- **Retail Security:** Automatic shoplifting detection in stores
- **Loss Prevention:** Real-time monitoring systems
- **Forensic Analysis:** Post-incident video review
- **Anomaly Detection:** Identifying suspicious behavior in surveillance footage
- **24/7 Monitoring:** Cost-effective continuous surveillance

## 🔬 Research & References

R3D-18 is based on:
- ResNet architecture (He et al., 2015)
- 3D CNN video classification (Tran et al., 2017)
- Kinetics dataset (Kay et al., 2017)

## 📝 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📧 Contact

For questions or inquiries, please reach out through the GitHub repository.

---

**Last Updated:** April 12, 2026