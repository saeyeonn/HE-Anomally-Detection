# Secure Sensor Data Interpolation and Anomaly Detection with Homomorphic Encryption

## 📋 Project Overview

This project implements a **privacy-preserving sensor data analysis system** using homomorphic encryption (HE) for industrial IoT environments. The system enables secure interpolation of missing sensor values and anomaly detection without exposing sensitive manufacturing data.

### 🎯 Key Features

- **Secure Data Processing**: Homomorphic encryption enables computation on encrypted data
- **Missing Value Interpolation**: Linear interpolation for handling sensor data gaps
- **Anomaly Detection**: Logistic regression model for identifying defective products
- **Web Interface**: User-friendly web application for data upload and analysis
- **Privacy Protection**: No decryption required during analysis phase

## 🏭 Problem Statement

Industrial sensor data contains critical information about:
- Production strategies and quality control intelligence
- Equipment usage patterns and proprietary manufacturing processes
- Line efficiency metrics and cost structure information

Traditional security methods (TLS, Secure Boot, Access Control) require decryption during analysis, creating security vulnerabilities. Our solution maintains data privacy throughout the entire analysis pipeline.

## 🔧 Technical Stack

### Core Technologies
- **Python 3.8+**: Main programming language
- **HEAAN**: Homomorphic encryption library
- **scikit-learn**: Machine learning algorithms
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing

### Web Framework
- **Flask**: Backend web framework
- **HTML/CSS/JavaScript**: Frontend interface
- **Bootstrap**: UI components

### Development Tools
- **Jupyter Notebook**: Data analysis and experimentation
- **Git**: Version control
- **Docker**: Containerization (optional)

## 📊 Dataset

The project uses the **SECOM Dataset** from UCI Machine Learning Repository:
- **1,567 samples** from semiconductor manufacturing process
- **590 numerical sensor measurements** (anonymized)
- **Binary classification**: Pass (normal) / Fail (defective product)
- **Missing values**: Naturally occurring NaN values for interpolation experiments

## 🚀 How to Run

### Prerequisites
```bash
# Install Python 3.8+
python --version

# Install pip
pip --version

# Install HeaaN stat sdk
docker pull cryptolabinc/heaan-stat:1.0.0-cpu
docker run -p 8888:8888 --rm -it cryptolabinc/heaan-stat:1.0.0-cpu
```

### Inside Docker Container
Once the container is running and Jupyter is accessible at http://localhost:8888:
bash# Open terminal in Jupyter or use docker exec

```bash
# Clone the repository
git clone https://github.com/saeyeonn/HE-Anomally-Detection.git
cd HE-Anomally-Detection

# Install additional dependencies if needed
pip install scikit-learn numpy 
pip install Flask==2.3.2
pip install flask-cors==3.0.10
pip install pandas==2.2.1
```

### Run the application
```bash
# Frontend
cd web/dist
python -m http.server 3000 # if 3000 port is unavailable, change to available port number

# Backend
cd server
python3 app.py # if python3 is unavailable, try 'python app.py'
```
### Go to the Web Page
Open a web browser and navigate to http://localhost:3000
Upload the file 'server/dataset/df_final_test.csv' to web page.


## 🧮 Algorithm Details

### Linear Interpolation (HE)
- **Input**: Encrypted sensor time series with missing values
- **Process**: Homomorphic linear interpolation using only `add`, `mult`, and `sub` operations
- **Output**: Complete encrypted time series

### Logistic Regression (HE)
- **Sigmoid Approximation**: Degree-3 Chebyshev polynomial: `σ(z) ≈ 0.5 + 0.25z - 0.0625z³`
- **Training**: Gradient descent with encrypted gradients
- **Prediction**: Encrypted probability scores for anomaly detection

## 📈 Results

### Performance Metrics
- **Plaintext Accuracy**: 89.17%
- **Encrypted Accuracy**: 68.33%
- **Privacy**: Complete data confidentiality maintained
- **Scalability**: Suitable for small to medium sensor networks

### Web Demo
- Upload CSV sensor data(./server/dataset/df_final_test.csv)
- Secure interpolation and analysis
- Timestamp-based anomaly results
- **Demo Video**: [YouTube Link](https://youtu.be/jL7xYZGqvxo?si=_KC1k35p9fg21Vjt)

## 📚 References

- [SECOM Dataset - UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/SECOM)
- [HEAAN Library](https://github.com/snucrypto/HEAAN)
- [Sensor Market Research](https://www.precedenceresearch.com/sensor-market)
- [Brightics Knowledge Sharing](https://www.brightics.ai/community/knowledge-sharing/detail/7059)

## 📞 Contact

For questions, please open an issue or contact the development team.

---

**Note**: This project was developed as part of an Information Security course focusing on privacy-preserving analytics for industrial IoT systems.