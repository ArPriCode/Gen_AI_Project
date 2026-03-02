# 📘 Intelligent Exam Question Analysis System

An advanced AI-powered analysis platform for evaluating and assessing educational exam questions using machine learning and psychometric metrics.

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-FF0000?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [API & Analysis Metrics](#api--analysis-metrics)
- [Data Format](#data-format)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

The **Intelligent Exam Question Analysis System** is a sophisticated machine learning solution designed to automatically evaluate the quality and difficulty of educational exam questions. It combines natural language processing, statistical analysis, and psychometric principles to provide comprehensive insights into question effectiveness.

### Key Capabilities
- 🤖 **ML-Powered Classification** - Uses trained logistic regression models for accuracy prediction
- 📊 **Psychometric Analysis** - Evaluates questions using educational science principles
- 📈 **Comprehensive Metrics** - Analyzes 15+ dimensions of question quality
- 🎓 **Bloom's Taxonomy Integration** - Classifies questions by cognitive complexity levels
- 💡 **Interactive Dashboard** - Streamlit-based user-friendly interface

---

## ✨ Features

### Core Analysis Features
| Feature | Description |
|---------|-------------|
| **Text Complexity Analysis** | TF-IDF vectorization for content difficulty assessment |
| **Readability Scoring** | Flesch Kincaid and other readability metrics |
| **Cognitive Level Classification** | Remember, Understand, Apply, Analyze, Evaluate, Create |
| **Learning Gap Detection** | Identifies areas where students struggle |
| **Discrimination Index** | Measures question's ability to differentiate students |
| **Question Difficulty Prediction** | Classifies as Easy, Medium, or Hard |
| **Assessment Quality Scoring** | Overall quality evaluation of the question |

### Dashboard Features
- **Real-time Analysis** - Instant question evaluation
- **Visual Insights** - Charts and visualizations of metrics
- **Batch Processing** - Analyze multiple questions efficiently
- **Export Capabilities** - Download analysis results in multiple formats
- **Subject-Specific Analysis** - Tailored evaluation for different subjects

---

## 🛠️ Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Framework** | Streamlit | Latest |
| **ML Library** | scikit-learn | 1.6.1 |
| **Data Processing** | pandas | 2.2.2 |
| **Numerical Computing** | NumPy | 2.0.2 |
| **Text Analysis** | textstat | Latest |
| **Model Serialization** | joblib | Latest |
| **Language** | Python | 3.8+ |

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step-by-Step Setup

1. **Clone or download the project**
```bash
cd /Users/arunkumargiri/Desktop/Gen_AI_Project
```

2. **Create a virtual environment** (optional but recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import streamlit; print(f'Streamlit {streamlit.__version__} installed successfully')"
```

---

## 🚀 Quick Start

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Basic Usage

1. **Input a Question** - Enter or paste an exam question in the interface
2. **Select Subject** - Choose the relevant subject (Mathematics, Physics, Computer Science, Engineering Aptitude)
3. **Analyze** - Click the analyze button to process
4. **Review Results** - Examine the metrics and recommendations
5. **Export** - Download the analysis results if needed

---

## 📂 Project Structure

```
Gen_AI_Project/
│
├── app.py                           # Main Streamlit application
├── GenAiProject.ipynb              # Jupyter notebook for model training/analysis
├── requirements.txt                 # Python dependencies
├── question_ans_analysis.csv       # Training/sample dataset
├── README.md                        # This file
│
└── Report/                         # Generated reports directory
    └── [Analysis outputs]
```

### File Descriptions

- **`app.py`** - Main application file containing the Streamlit interface and analysis logic
- **`GenAiProject.ipynb`** - Jupyter notebook with exploratory analysis, model training, and evaluation
- **`question_ans_analysis.csv`** - Dataset containing question features and labels for training
- **`.pkl` files** - Pre-trained machine learning models
  - `lr_model.pkl` - Logistic Regression classifier
  - `tfidf.pkl` - TF-IDF vectorizer
  - `scaler.pkl` - Feature scaler

---

## 🧠 How It Works

### Analysis Pipeline

```
Question Input
    ↓
Text Cleaning & Preprocessing
    ↓
Feature Extraction
  ├─ TF-IDF Vectorization
  ├─ Readability Metrics
  ├─ Structural Features
  └─ Linguistic Features
    ↓
Model Prediction
  ├─ Logistic Regression Classification
  └─ Confidence Scoring
    ↓
Result Aggregation & Presentation
    ↓
Comprehensive Analysis Reports
```

### Key Processing Steps

1. **Text Preprocessing**
   - Convert to lowercase
   - Remove special characters
   - Normalize whitespace
   - Tokenization

2. **Feature Extraction**
   - TF-IDF vector generation (text complexity)
   - Word count and sentence count
   - Readability scores (Flesch Reading Ease, etc.)
   - Subject metadata extraction

3. **Model Inference**
   - Input scaling using pre-fitted scaler
   - Logistic regression prediction
   - Confidence score calculation
   - Difficulty classification

4. **Results Presentation**
   - Metric aggregation
   - Visual representation
   - Recommendation generation
   - Export formatting

---

## 📊 API & Analysis Metrics

### Input Features

| Feature | Type | Description |
|---------|------|-------------|
| `question_text` | string | The exam question content |
| `subject` | string | Subject area (Mathematics, Physics, CS, etc.) |
| `cognitive_level_bloom` | string | Bloom's taxonomy level |
| `readability_score` | float | Text readability metric (0-100) |
| `word_count` | int | Number of words in question |
| `sentence_count` | int | Number of sentences |

### Output Metrics

| Metric | Range | Description |
|--------|-------|-------------|
| `difficulty_label` | Easy/Medium/Hard | Predicted difficulty level |
| `assessment_quality_score` | 0.0-1.0 | Overall question quality (higher is better) |
| `learning_gap_score` | 0.0-1.0 | Identifies learning gaps |
| `discrimination_index` | 0.0-1.0 | Question's ability to differentiate students |
| `correct_percentage` | 0.0-1.0 | Predicted student success rate |
| `readability_score` | 0.0-100 | Text complexity for student comprehension |

---

## 📋 Data Format

### Input CSV Format

The training data should follow this structure:

```csv
question_text,subject,cognitive_level_bloom,readability_score,word_count,sentence_count,time_taken_minutes,total_students_attempted,correct_attempts,incorrect_attempts,correct_percentage,learning_gap_score,discrimination_index,difficulty_label,assessment_quality_score
```

### Sample Data

| Question | Subject | Bloom Level | Readability | Difficulty |
|----------|---------|-------------|-------------|------------|
| Solve the quadratic equation... | Mathematics | create | 77.49 | hard |
| Implement binary search... | Computer Science | understand | 45.01 | easy |
| Apply Newton's law... | Physics | create | 89.84 | hard |

---

## ⚙️ Configuration

### Application Settings

Edit `app.py` to customize:

```python
# Page Configuration
st.set_page_config(
    page_title="Intelligent Exam Question Analysis",
    page_icon="📘",
    layout="wide"
)

# Supported Subjects
SUBJECTS = [
    'Mathematics',
    'Physics',
    'Computer Science',
    'Engineering Aptitude'
]

# Bloom's Taxonomy Levels
BLOOM_LEVELS = [
    'remember',
    'understand',
    'apply',
    'analyze',
    'evaluate',
    'create'
]
```

### Model Configuration

- **Algorithm**: Logistic Regression
- **Vectorizer**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Feature Scaling**: StandardScaler
- **Max features**: Configurable in model training

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Module Not Found Errors
```bash
# Solution: Reinstall dependencies
pip install --upgrade -r requirements.txt
```

#### 2. Model Files Missing
```bash
# Solution: Ensure .pkl files are in the project root
ls -la *.pkl
```

#### 3. Streamlit Page Not Loading
```bash
# Solution: Clear cache and restart
streamlit cache clear
streamlit run app.py
```

#### 4. Memory Issues with Large CSV
```python
# Use chunking in data processing
chunks = pd.read_csv('data.csv', chunksize=1000)
```

### Performance Optimization

- **Caching**: Streamlit automatically caches model loading with `@st.cache_resource`
- **Batch Processing**: Process multiple questions in parallel
- **GPU Acceleration**: For large datasets, consider GPU-enabled scikit-learn

---

## 👥 Contributing

We welcome contributions! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add docstrings to functions
- Include unit tests for new features
- Update documentation



## 📞 Support & Contact

For issues, questions, or suggestions:
- Open an issue on the repository
- Contact the development team
- Check documentation and examples

---

## 🔍 Additional Resources

### Related Technologies
- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Bloom's Taxonomy](https://en.wikipedia.org/wiki/Bloom%27s_taxonomy)
- [TF-IDF Overview](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

### Papers & References
- Educational assessment best practices
- Machine learning for educational data
- Psychometric analysis methodologies

---

**Last Updated**: March 2, 2026  
**Version**: 1.0.0  
**Maintainer**: Gen AI Team
