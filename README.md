# TrueNewsAI

**AAI-540 MLOps, Group 1**  
**Members:** Matthew Guzman, Isaack Karanja, and Jeremy Krick

---

## Project Overview

TrueNewsAI is an end-to-end Machine Learning Operations (MLOps) project that builds, trains, and deploys a deep learning model for fake news detection. The project demonstrates best practices in model development, versioning, monitoring, and cloud deployment using modern ML tools and infrastructure.

### Problem Statement

With the proliferation of misinformation online, there is a critical need for automated systems that can reliably distinguish between fake and real news articles. This project develops a BERT-based transformer model to classify news articles as genuine or fabricated based on their text content.

**Data Source:** [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/data)

---

## Project Goals

1. **Model Development:** Build a high-accuracy binary classification model using state-of-the-art transformer architectures (BERT)
2. **MLOps Pipeline:** Implement a complete ML lifecycle including training, evaluation, versioning, and deployment
3. **Model Registry:** Publish trained models to HuggingFace Model Hub for easy sharing and deployment
4. **Monitoring & Observability:** Implement monitoring to track model performance in production
5. **Cloud Integration:** Leverage AWS services (S3, SageMaker) for scalable training and inference

---

## Architecture Overview

### High-Level Workflow

```
Data Preparation → Model Training → Model Evaluation → Model Registry → Deployment & Monitoring
```

### Components

#### 1. **Data Pipeline**
- **Source:** Kaggle fake/real news dataset
- **Storage:** AWS S3 for centralized data management
- **Processing:** Text preprocessing including stopword removal
- **Format:** HuggingFace Datasets format for seamless integration with transformers

#### 2. **Model Training**
- **Model Architecture:** BERT-based transformer for sequence classification (binary: fake/real)
- **Framework:** HuggingFace Transformers + PyTorch
- **Training Script:** `scripts/train.py`
  - Uses HuggingFace Trainer API for efficient training
  - Supports distributed training on AWS SageMaker
  - Implements evaluation metrics (accuracy) during training
  - Saves checkpoints and best model to S3

#### 3. **Model Inference**
- **Inference Script:** `scripts/inference.py`
- **Deployment:** AWS SageMaker endpoint for scalable REST API
- **Handler:** Custom SageMaker inference handler for HuggingFace models
- **Input/Output:** JSON-based API for predictions

#### 4. **Experiment Notebooks**
- `train and publish using huggingface.ipynb` - End-to-end training and publishing workflow
- `train and publish using huggingface_monitoring.ipynb` - Training with monitoring capabilities
- `Train BERT and Save Model to Model Registry.ipynb` - BERT model training and versioning
- `Update news dataset to S3.ipynb` - Data pipeline and S3 upload automation

---

## Technical Stack

- **Model Framework:** PyTorch + HuggingFace Transformers
- **Training Platform:** AWS SageMaker
- **Model Registry:** HuggingFace Model Hub
- **Data Storage:** AWS S3
- **Data Format:** HuggingFace Datasets
- **Language:** Python 3.x
- **Monitoring:** CloudWatch (via SageMaker)

---

## Project Results

### Model Performance
- **Task:** Binary classification (Fake vs. Real news)
- **Metric:** Accuracy on test set
- **Training Configuration:**
  - Epochs: 3
  - Batch Size: 32 (train) / 64 (eval)
  - Learning Rate: 5e-5
  - Warmup Steps: 500
  - Loss: Cross-entropy

### Key Milestones
- ✅ Data pipeline setup and S3 integration
- ✅ BERT model training and fine-tuning
- ✅ HuggingFace Model Hub publishing
- ✅ SageMaker inference endpoint deployment
- ✅ Model monitoring and performance tracking

### Artifacts
- Trained BERT model available on HuggingFace Model Hub
- Evaluation results and training metrics stored in S3
- Deployment-ready SageMaker model artifacts

---

## How to Use

### Training
```bash
python scripts/train.py \
  --model_id "bert-base-uncased" \
  --epochs 3 \
  --train_batch_size 32 \
  --eval_batch_size 64
```

### Inference
```bash
python scripts/inference.py
```

### Interactive Development
See Jupyter notebooks in the project root for end-to-end workflows including data preparation, training, and deployment.

---

## Project Structure

```
TrueNewsAI/
├── README.md                                          # This file
├── scripts/
│   ├── train.py                                      # SageMaker training script
│   └── inference.py                                  # SageMaker inference handler
├── data/                                             # Local data directory
├── train and publish using huggingface.ipynb        # Main training workflow
├── train and publish using huggingface_monitoring.ipynb  # Training with monitoring
├── Train BERT and Save Model to Model Registry.ipynb # BERT fine-tuning
└── Update news dataset to S3.ipynb                  # Data pipeline
```

---

## Next Steps & Future Improvements

- [ ] Improve model accuracy through hyperparameter tuning
- [ ] Expand to multi-class classification (satire, propaganda, etc.)
- [ ] Implement A/B testing framework for model versions
- [ ] Add data drift detection and monitoring
- [ ] Create automated retraining pipeline
- [ ] Build web UI for model predictions
- [ ] Add explainability features (attention visualization, SHAP)
