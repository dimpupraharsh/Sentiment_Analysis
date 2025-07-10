# 🧠 Emotion Classification using BERT: A Comparative Study of Fine-Tuning Strategies

![Model Visualization](https://your-repo-path/Screenshot-Model-Results.png)

## 📌 Project Overview

This repository presents a comparative study of **three fine-tuning techniques** applied to a pre-trained `bert-base-uncased` model for **multi-class emotion classification**. Using the [SetFit/emotion](https://huggingface.co/datasets/SetFit/emotion) dataset, we explore the performance of:

- ✅ **Head-based tuning** (MLP classifier)
- 🔄 **Adapter-based tuning** (Pfeiffer-style bottleneck adapters)
- 🧠 **Full fine-tuning** (All 110M BERT parameters updated)

---

## 📂 Table of Contents

1. [📊 Motivation & Objective](#-motivation--objective)  
2. [🧠 Model Architectures](#-model-architectures)  
3. [📚 Dataset Details](#-dataset-details)  
4. [⚙️ Methodology](#-methodology)  
5. [📈 Results & Evaluation](#-results--evaluation)  
6. [📋 Sample Predictions](#-sample-predictions)  
7. [🛠️ Setup Instructions](#-setup-instructions)  
8. [📎 References](#-references)  

---

## 📊 Motivation & Objective

Fine-tuning transformer-based models can be computationally expensive. This project investigates **parameter-efficient alternatives** like adapter tuning and head-based tuning, and benchmarks them against full fine-tuning to strike a balance between accuracy and efficiency.

---

## 🧠 Model Architectures

| Strategy | Description | Trainable Params |
|----------|-------------|------------------|
| **Head-based** | Freeze all BERT weights, append MLP classifier | ~1M |
| **Adapter-based** | Insert adapter modules into transformer layers | ~0.5M |
| **Full Fine-tuning** | Update all BERT parameters | ~110M |

> 🔍 All variants are based on the [`bert-base-uncased`](https://huggingface.co/bert-base-uncased) model.

---

## 📚 Dataset Details

- 📦 **Source**: [SetFit/emotion](https://huggingface.co/datasets/SetFit/emotion)
- 🧾 **Labels**: `sadness`, `joy`, `love`, `anger`, `fear`, `surprise`
- 📊 **Split**:
  - Train: ~16,000
  - Validation: ~2,000
  - Test: ~2,000

✅ The dataset is relatively balanced across the six emotions.

---

## ⚙️ Methodology

### 🔁 Preprocessing
- Tokenization with `BertTokenizer` (max length: 128)
- [CLS] and [SEP] tokens added
- Padding and truncation applied

### 🧪 Training Settings
- Optimizer: `AdamW`
- Learning Rate: `1e-5`
- Loss Function: `CrossEntropyLoss`
- Epochs: `10`
- Batch Size: 32 (train), 64 (eval)

### 🛠️ Implementation Framework
- Hugging Face Transformers
- Adapter-Transformers
- PyTorch
- scikit-learn for metrics

---

## 📈 Results & Evaluation

### 🔻 Training and Validation Loss

![Loss Plot](./plots/loss_comparison.png)

📌 **Observation**: Full fine-tuning converges fastest. Adapter tuning shows consistent reduction. MLP training is slowest due to frozen encoder.

---

### 📈 Validation Accuracy

![Accuracy Plot](./plots/val_accuracy.png)

📌 **Observation**:  
- Full: **94.2%**  
- Adapter: **92.9%**  
- MLP: **53.3%**

---

### 🧪 Test Set Metrics

![Test Metrics Plot](./plots/test_metrics.png)

| Metric     | MLP   | Adapter | Full |
|------------|-------|---------|------|
| Accuracy   | 0.54  | 0.91    | 0.93 |
| F1 Score   | 0.45  | 0.92    | 0.93 |
| Precision  | 0.52  | 0.92    | 0.93 |
| Recall     | 0.56  | 0.91    | 0.93 |

📌 **Insight**: Adapter tuning provides near-full performance with significantly fewer trainable parameters.

---

## 📋 Sample Predictions

```
Text: i pay attention it deepens into a feeling of being invaded and helpless
True Emotion: fear
Predicted Emotion: fear

Text: i feel extremely comfortable with the group of people that i dont even need to hide myself
True Emotion: joy
Predicted Emotion: joy

Text: i find myself in the odd position of feeling supportive of
True Emotion: love
Predicted Emotion: love
```

✅ Full fine-tuned model accurately captures nuanced emotional cues even in long or abstract text.

---

## 🛠️ Setup Instructions

### 🔧 Environment

```bash
pip install transformers
pip install datasets
pip install adapter-transformers
pip install scikit-learn
pip install matplotlib seaborn
```

### ▶️ Run Training

```bash
python train.py --mode full       # for full fine-tuning
python train.py --mode adapter    # for adapter tuning
python train.py --mode mlp        # for head-based MLP
```

### 📊 Evaluate

```bash
python evaluate.py --model_path ./saved_model/full
```

---

## 📎 References

- Devlin et al. (2018) [BERT](https://arxiv.org/abs/1810.04805)
- Vaswani et al. (2017) [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- Liu et al. (2019) [RoBERTa](https://arxiv.org/abs/1907.11692)
- Raffel et al. (2020) [T5](https://arxiv.org/abs/1910.10683)
- Stickland & Murray (2019) [PALs](https://arxiv.org/abs/1902.02671)
- Howard & Ruder (2018) [ULMFiT](https://arxiv.org/abs/1801.06146)
- Houlsby et al. (2019) [Adapters](https://arxiv.org/abs/1902.00751)
- Pfeiffer et al. (2020) [AdapterHub](https://arxiv.org/abs/2007.07779)
- Mosbach et al. (2021) [Fine-tuning Stability](https://arxiv.org/abs/2006.04884)

---

## 👨‍💻 Author

**Praharsh Vijay**  
ID: 22098361  
MSc Data Science (Advanced Research), University of Hertfordshire
---

## 📊 Detailed Epoch-wise Metrics

### 🔵 Head-Based Tuning (MLP Classifier)

| Epoch | Train Loss | Val Loss | Accuracy | F1 Score | Recall | Precision |
|-------|------------|----------|----------|----------|--------|-----------|
| 1     | 1.5220     | 1.4249   | 0.4955   | 0.3822   | 0.4955 | 0.3111    |
| 2     | 1.4089     | 1.3459   | 0.5000   | 0.3885   | 0.5000 | 0.4235    |
| 3     | 1.3571     | 1.3121   | 0.5120   | 0.4034   | 0.5120 | 0.4814    |
| 4     | 1.3300     | 1.2885   | 0.5145   | 0.4144   | 0.5145 | 0.5608    |
| 5     | 1.3129     | 1.2664   | 0.5235   | 0.4319   | 0.5235 | 0.5529    |
| 6     | 1.2998     | 1.2590   | 0.5260   | 0.4429   | 0.5260 | 0.5185    |
| 7     | 1.2896     | 1.2499   | 0.5285   | 0.4492   | 0.5285 | 0.5487    |
| 8     | 1.2805     | 1.2448   | 0.5305   | 0.4525   | 0.5305 | 0.5468    |
| 9     | 1.2789     | 1.2413   | 0.5335   | 0.4569   | 0.5335 | 0.5166    |
| 10    | 1.2778     | 1.2399   | 0.5335   | 0.4570   | 0.5335 | 0.5141    |

**Test Classification Report:**
- Accuracy: 0.5415
- Macro F1: 0.2855
- Weighted F1: 0.4646
- Strongest class performance: `joy`, `sadness`
- Weakest: `surprise`, `love`

---

### 🟠 Adapter-Based Tuning

| Epoch | Train Loss | Val Loss | Accuracy | F1 Score | Recall | Precision |
|-------|------------|----------|----------|----------|--------|-----------|
| 1     | 0.9864     | 0.5410   | 0.8060   | 0.8024   | 0.8060 | 0.8031    |
| 2     | 0.4410     | 0.3180   | 0.8915   | 0.8904   | 0.8915 | 0.8915    |
| 3     | 0.2920     | 0.2443   | 0.9135   | 0.9133   | 0.9135 | 0.9132    |
| 4     | 0.2377     | 0.2324   | 0.9175   | 0.9179   | 0.9175 | 0.9195    |
| 5     | 0.2022     | 0.2040   | 0.9225   | 0.9219   | 0.9225 | 0.9230    |
| 6     | 0.1805     | 0.1935   | 0.9240   | 0.9239   | 0.9240 | 0.9242    |
| 7     | 0.1736     | 0.1931   | 0.9275   | 0.9262   | 0.9275 | 0.9278    |
| 8     | 0.1576     | 0.1799   | 0.9245   | 0.9248   | 0.9245 | 0.9253    |
| 9     | 0.1537     | 0.1779   | 0.9295   | 0.9295   | 0.9295 | 0.9298    |
| 10    | 0.1497     | 0.1742   | 0.9245   | 0.9246   | 0.9245 | 0.9252    |

**Test Classification Report:**
- Accuracy: 0.9185
- Macro F1: 0.8704
- Weighted F1: 0.9178
- All emotion classes show strong balance

---

### 🟢 Full Fine-Tuning

| Epoch | Train Loss | Val Loss | Accuracy | F1 Score | Recall | Precision |
|-------|------------|----------|----------|----------|--------|-----------|
| 1     | 0.4119     | 0.1689   | 0.9305   | 0.9311   | 0.9305 | 0.9348    |
| 2     | 0.1320     | 0.1454   | 0.9355   | 0.9342   | 0.9355 | 0.9372    |
| 3     | 0.1069     | 0.1474   | 0.9415   | 0.9415   | 0.9415 | 0.9421    |
| 4     | 0.0836     | 0.1582   | 0.9400   | 0.9407   | 0.9400 | 0.9436    |
| 5     | 0.0655     | 0.2348   | 0.9370   | 0.9371   | 0.9370 | 0.9380    |
| 6     | 0.0419     | 0.2056   | 0.9410   | 0.9407   | 0.9410 | 0.9409    |
| 7     | 0.0248     | 0.2757   | 0.9390   | 0.9390   | 0.9390 | 0.9392    |
| 8     | 0.0126     | 0.3228   | 0.9425   | 0.9426   | 0.9425 | 0.9429    |
| 9     | 0.0102     | 0.3162   | 0.9410   | 0.9412   | 0.9410 | 0.9417    |
| 10    | 0.0059     | 0.3267   | 0.9420   | 0.9421   | 0.9420 | 0.9423    |

**Test Classification Report:**
- Accuracy: 0.9305
- Macro F1: 0.8871
- Weighted F1: 0.9306
- Outstanding precision and recall across all labels

---
