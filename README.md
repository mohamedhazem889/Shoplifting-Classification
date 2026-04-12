# Shoplifting Classification

A deep learning project for detecting and classifying shoplifting behavior in surveillance videos using 3D Convolutional Neural Networks (3D CNN).

## 📋 Project Overview

This repository contains an implementation of a shoplifting detection system that leverages video analysis and deep learning techniques. The project uses 3D CNN models to process video sequences and classify whether shoplifting activities are occurring in surveillance footage.

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
- **Deep Learning Framework:** TensorFlow/Keras
- **Model Architecture:** 3D Convolutional Neural Networks (3D CNN)
- **Containerization:** Docker & Docker Compose
- **Framework:** Django (Web API)

## 📁 Repository Structure

```
├── Model_3dCNN.ipynb          # Main 3D CNN model implementation and training
├── pretrained.ipynb           # Pre-trained model inference and evaluation
├── shoplifting_api/           # Django API for model serving
├── detector/                  # Detection utilities and helper modules
├── templates/                 # HTML templates for web interface
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration for containerization
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
jupyter notebook Model_3dCNN.ipynb
jupyter notebook pretrained.ipynb
```

## 📊 Model Architecture

The project implements a **3D Convolutional Neural Network (3D CNN)** that:
- Processes video sequences as 3D tensors (height × width × frames)
- Extracts spatial and temporal features from video data
- Classifies video segments as shoplifting or normal behavior
- Achieves robust performance on surveillance video data

### Key Features:
- **Temporal Modeling:** Captures motion and behavior patterns across frames
- **Spatial Feature Extraction:** Detects objects and activities within frames
- **Real-time Inference:** Optimized for practical deployment

## 🔧 API Usage

The project includes a Django-based REST API for model inference:

```bash
python manage.py runserver
```

Access the API endpoints for:
- Video upload and processing
- Real-time shoplifting detection
- Model predictions and confidence scores

## 📈 Model Training

The `Model_3dCNN.ipynb` notebook includes:
- Data preprocessing and augmentation
- Model architecture definition
- Training pipeline with validation
- Performance metrics and visualizations
- Model checkpointing and saving

## 🎯 Use Cases

- **Retail Security:** Automatic shoplifting detection in stores
- **Loss Prevention:** Real-time monitoring systems
- **Forensic Analysis:** Post-incident video review
- **Anomaly Detection:** Identifying suspicious behavior in surveillance footage

## 📝 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📧 Contact

For questions or inquiries, please reach out through the GitHub repository.

---

**Last Updated:** April 12, 2026