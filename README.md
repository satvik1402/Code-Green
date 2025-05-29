# DSA Plagiarism Detector

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0.1-blue.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated plagiarism detection system specifically designed for Data Structures and Algorithms (DSA) code submissions. This tool helps educators and coding competition organizers identify potential cases of code plagiarism while being smart enough to recognize legitimate code similarities.

## ✨ Features

- **Multiple Detection Techniques**
  - Exact match detection
  - Variable renaming detection
  - Structural similarity analysis
  - Comment similarity analysis
  - Semantic analysis using CodeBERT
  - Copy paste detection by fast typing detection

- **Smart Algorithm Recognition**
  - Identifies common DSA patterns 
  - Recognizes different implementations of the same algorithm
  - Detects refactored code with high accuracy

- **Comprehensive Reporting**
  - Detailed similarity scores
  - Visual charts and metrics
  - Side-by-side code comparison
  - Suspicious code highlighting

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/dsa-plagiarism-detector.git
   cd dsa-plagiarism-detector
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the database**
   ```bash
   python -m CodePlagueDetector.db_migrate
   ```

### Running the Application

1. **Start the development server**
   ```bash
   python -m CodePlagueDetector.main
   ```

2. **Access the application**
   Open your browser and navigate to: http://localhost:5000

## 🛠️ How It Works

### Detection Methods

1. **Exact Match Detection**
   - Compares code character-by-character
   - Ignores whitespace and comments
   - Identifies direct copies of reference solutions

2. **Variable Renaming Detection**
   - Normalizes variable and function names
   - Compares abstract syntax trees (ASTs)
   - Identifies structurally similar code with renamed identifiers

3. **Structural Similarity**
   - Analyzes code structure and control flow
   - Recognizes common algorithm patterns
   - Detects similar logic with different implementations

4. **Semantic Analysis**
   - Uses CodeBERT for deep code understanding
   - Identifies semantically similar code
   - Detects paraphrased solutions

### Architecture

```
DSA-Plagiarism-Detector/
├── CodePlagueDetector/
│   ├── __init__.py      # Application factory
│   ├── main.py          # Entry point
│   ├── models.py        # Database models
│   ├── routes.py        # URL routes and views
│   ├── plagiarism_detector.py  # Core detection logic
│   └── dsa_problems.py  # Problem definitions
├── tests/               # Test files
├── static/              # Static files (CSS, JS, images)
├── templates/           # HTML templates
├── app.py               # Main application script
└── requirements.txt     # Dependencies
```

## 📊 Example Usage

1. Select a DSA problem from the dashboard
2. Write or paste your solution in the code editor
3. Click "Analyze for Plagiarism"
4. View detailed analysis including:
   - Originality score
   - Similarity metrics
   - Detected patterns
   - Comparison with reference solutions

## 📝 Adding New Problems

1. Edit `CodePlagueDetector/dsa_problems.py`
2. Add a new problem definition:
   ```python
   {
       'id': 'unique_problem_id',
       'title': 'Problem Title',
       'description': 'Detailed problem description',
       'difficulty': 'Easy/Medium/Hard',
       'function_signature': 'def solution_name(parameters):',
       'test_cases': [
           {'input': 'input1', 'output': 'expected1'},
           # Add more test cases
       ]
   }
   ```
## Contributors-
- Kanha gupta
- Rahul Shekhawat
- Tushar Singhal