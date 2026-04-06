import csv
import logging
from datetime import datetime
from pathlib import Path
import numpy as np

class PredictionLogger:
    """Structured logging for predictions and model metrics."""
    
    def __init__(self, log_file='logs/predictions.csv', log_level='INFO'):
        self.log_file = log_file
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Setup Python logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV file with headers if not exists."""
        if not Path(self.log_file).exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'image_path', 'prediction', 'confidence',
                    'is_uncertain', 'model_version', 'processing_time_ms'
                ])
    
    def log_prediction(self, image_path, prediction, confidence, 
                      processing_time_ms, model_version="v1"):
        """Log a prediction to CSV and console."""
        is_uncertain = confidence < 0.6
        
        # Log to console
        self.logger.info(
            f"Prediction: {prediction} | Confidence: {confidence:.3f} | "
            f"Uncertain: {is_uncertain} | Image: {image_path}"
        )
        
        # Log to CSV
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                str(image_path),
                prediction,
                f"{confidence:.4f}",
                is_uncertain,
                model_version,
                f"{processing_time_ms:.2f}"
            ])
    
    def get_metrics(self):
        """Calculate metrics from logged predictions."""
        metrics = {
            'total_predictions': 0,
            'uncertain_predictions': 0,
            'average_confidence': 0,
            'shoplifting_predictions': 0
        }
        
        confidences = []
        try:
            with open(self.log_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    metrics['total_predictions'] += 1
                    confidence = float(row['confidence'])
                    confidences.append(confidence)
                    
                    if row['is_uncertain'] == 'True':
                        metrics['uncertain_predictions'] += 1
                    if row['prediction'] == 'Shoplifting':
                        metrics['shoplifting_predictions'] += 1
        except FileNotFoundError:
            return metrics
        
        if confidences:
            metrics['average_confidence'] = np.mean(confidences)
        
        return metrics
