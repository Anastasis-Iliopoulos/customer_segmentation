# **Customer Segmentation**

## **Overview**
Customer segmentation with KMeans clustering employing MLFlow.

---

> **Important Note:** Make sure that the `Data.zip` file is unzipped.

## **Files**
```bash
.
├── create_and_save_features.ipynb
├── EDA_features_data.ipynb
├── EDA_raw_data_1.ipynb
├── EDA_raw_data_2.ipynb
├── EDA_results.ipynb
├── Experimentation.ipynb
├── feature_engineering_test.ipynb
├── evaluation.py
├── feature_engineering.py
├── inference.py
├── train_model.py
├── transformations.py
├── utils.py
├── config.yaml
├── requirements.txt
├── logs.log
├── README.md
```

---

## **Setup Instructions**

### **Prerequisites**
- Python 3.7+
- Recommended: A virtual environment for dependencies.

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/Anastasis-Iliopoulos/customer-segmentation
   cd customer-segmentation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure `config.yaml` for paths and parameters.

---

## **How to Run**

### **1. Data Preprocessing and Feature Engineering**
feature_engineering.py (cli support isn't implemented)

### **2. Train the Model**
Train the KMeans model and log results to MLFlow:
```bash
python -m scripts.train_model --config config.yaml
```

### **3. Inference**
Run the inference pipeline to assign cluster labels to new data:
```bash
python -m scripts.inference --run_id <mlflow-run-id> --input_data <path-to-input-parquet> --output_data <path-to-output-parquet>
```

(Register model to mlflow registry - for serving - hasn't implemented)

---

## **File Descriptions**

### **Notebooks**
- ~~**load_and_schema_checks_spark.ipynb**: pyspark test~~
- **EDA_raw_data_1.ipynb**: Analysis on raw data
- **EDA_raw_data_2.ipynb**: More analysis on raw data
- **create_and_save_features.ipynb**: create and save features
- ~~**feature_engineering_test.ipynb**: test~~
- **EDA_features_data.ipynb**: Analysis on features.
- **Experimentation.ipynb**: Prototyping and hyperparameter tuning for the KMeans model.
- **EDA_results.ipynb**: Final results with n_cluster=3 (and an experiment with n_cluster=5)


### **Scripts**
- **`feature_engineering.py`**: Functions to compute and save customer-level features.
- **`train_model.py`**: Orchestrates the training pipeline and logs artifacts to MLFlow.
- **`inference.py`**: Loads trained models and applies clustering to new data.
- **`transformations.py`**: Implements reusable preprocessing steps.
- **`evaluation.py`**: Computes metrics like silhouette scores for clustering.
- **`utils.py`**: Helper functions for logging, saving/loading data etc.

### **Configuration**
- **`config.yaml`**: configuration for paths, parameters, and pipeline setup. (not everything is implemented)

### **Dependencies**
- **`requirements.txt`**: Lists Python packages required for the project.

---

## **Evaluation**
- Silhouette Score
- Inertia

---

## **Future Enhancements**
0. better structure: 
```bash
.
├── customer_segmentation/
│   ├── main_or_handler.py
│   ├── __init__.py
│   ├── core/
│   │   ├── evaluation.py
│   │   ├── feature_engineering.py
│   │   ├── inference.py
│   │   ├── train_model.py
│   │   ├── transformations.py
│   │   ├── utils.py
│   │   └── __init__.py
│
├── Data/
│   ├── logs/
│   │   └── logs.log
│   ├── raw/
│   │   ├── order.csv
│   │   └── product.csv
│   ├── results/
│   │   ├── clustered_3.csv
│   │   ├── clustered_3_b1dffefa3ad844e3bc3bc867199845ff.parquet.gzip
│   │   └── consumers_features.parquet.gzip
│   └── sample/
│       ├── order.csv
│       └── product.csv
│
├── notebooks/
│   ├── create_and_save_features.ipynb
│   ├── EDA_features_data.ipynb
│   ├── EDA_raw_data_1.ipynb
│   ├── EDA_raw_data_2.ipynb
│   ├── EDA_results.ipynb
│   ├── Experimentation.ipynb
│   ├── feature_engineering_test.ipynb
│   └── load_and_schema_checks_spark.ipynb
│
├── .gitignore
├── config.yaml
├── README.md
├── requirements.txt
└── setup.py
```
1. Use databases and/or cloud storage (like AWS S3) instead of files and local filesystem.
1. Use FastAPI to create an app for serving the model.
1. Alterntively use AWS Lambda functions for serving the model.
1. Setup Jenkins for deploying.
1. Produce documentation with Sphinx and/or Swagger.
1. Use Airflow and/or Databricks to orchestrate pipelines, process the data, training, experimentation etc (also use other features like feature store). Change the code to make use of pyspark. 
1. Experiment with other clustering algorithms like DBSCAN.
1. Add mechanism for monitoring training and inference.
