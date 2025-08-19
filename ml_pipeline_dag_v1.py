from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from airflow.operators.python import PythonOperator
import logging
import pandas as pd
from google.cloud import bigquery, storage
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json
import os

# Configuration - YOUR PROJECT DETAILS
PROJECT_ID = "flawless-haven-468319-k2"  # Your exact project ID
DATASET_ID = "ml_dataset"
TABLE_ID = "bike_features"
BUCKET_NAME = f"{PROJECT_ID}-ml-models"

# Default arguments
default_args = {
    'owner': 'data-scientist',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'ml_bike_prediction_pipeline',
    default_args=default_args,
    description='ML Pipeline for London Bike Trip Duration Prediction',
    schedule_interval='0 0 * * 0',  # Weekly on Sunday
    catchup=False,
    tags=['ml', 'bigquery', 'classification'],
)

# Task 1: Data Extraction
create_dataset_job = {
    "query": f"""
    CREATE SCHEMA IF NOT EXISTS `{PROJECT_ID}.{DATASET_ID}`
    """,
    "use_legacy_sql": False,
}

extract_data_job = {
    "query": f"""
    CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}` AS
    SELECT
      rental_id,
      duration,
      CASE 
        WHEN duration <= 900 THEN 'short'      -- <= 15 minutes
        WHEN duration <= 3600 THEN 'medium'    -- <= 1 hour
        ELSE 'long'                            -- > 1 hour
      END as duration_category,
      
      -- Time features
      EXTRACT(HOUR FROM start_date) as start_hour,
      EXTRACT(DAYOFWEEK FROM start_date) as day_of_week,
      EXTRACT(MONTH FROM start_date) as month,
      
      -- Station features
      start_station_id,
      end_station_id,
      
      -- Calculate if same station return
      CASE WHEN start_station_id = end_station_id THEN 1 ELSE 0 END as same_station,
      
    FROM `bigquery-public-data.london_bicycles.cycle_hire`
    WHERE 
      duration > 60  -- Remove very short trips (likely errors)
      AND duration < 86400  -- Remove trips longer than 24 hours
      AND start_date >= '2022-01-01'  -- Use recent data
      AND start_date < '2023-01-01'
      AND start_station_id IS NOT NULL
      AND end_station_id IS NOT NULL
    LIMIT 50000  -- Keep dataset manageable for demo
    """,
    "use_legacy_sql": False,
}

create_dataset_task = BigQueryInsertJobOperator(
    task_id='create_dataset',
    configuration={"query": create_dataset_job},
    dag=dag,
)

extract_data_task = BigQueryInsertJobOperator(
    task_id='extract_bike_data',
    configuration={"query": extract_data_job},
    dag=dag,
)

# Task 2: Model Training
def train_model(**context):
    """Train and save ML model with comprehensive logging"""
    
    # Initialize clients
    client = bigquery.Client(project=PROJECT_ID)
    storage_client = storage.Client(project=PROJECT_ID)
    
    # Load data from BigQuery
    query = f"""
    SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
    """
    
    logging.info("🔄 Loading data from BigQuery...")
    df = client.query(query).to_dataframe()
    logging.info(f"✅ Loaded {len(df)} rows from BigQuery")
    
    # Data preprocessing
    feature_columns = ['start_hour', 'day_of_week', 'month', 'start_station_id', 'end_station_id', 'same_station']
    X = df[feature_columns].copy()
    y = df['duration_category']
    
    # Handle categorical encoding for station IDs (keep top 100 stations, others as 'other')
    logging.info("🔄 Preprocessing station data...")
    for col in ['start_station_id', 'end_station_id']:
        top_stations = X[col].value_counts().head(100).index
        X[col] = X[col].apply(lambda x: x if x in top_stations else 'other')
        
    # Encode categorical variables
    label_encoders = {}
    for col in ['start_station_id', 'end_station_id']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logging.info(f"📊 Training set: {len(X_train)} samples")
    logging.info(f"📊 Test set: {len(X_test)} samples")
    
    # Train model
    logging.info("🤖 Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    logging.info(f"🎯 Model Accuracy: {accuracy:.4f}")
    logging.info(f"📈 Classification Report:\n{classification_report(y_test, y_pred)}")
    
    # Create timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model to GCS
    logging.info("💾 Saving model to Cloud Storage...")
    model_filename = f"bike_duration_model_{timestamp}.joblib"
    local_model_path = f"/tmp/{model_filename}"
    joblib.dump(model, local_model_path)
    
    # Upload model to GCS
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"models/{model_filename}")
    blob.upload_from_filename(local_model_path)
    
    # Save label encoders
    encoders_filename = f"label_encoders_{timestamp}.joblib"
    local_encoders_path = f"/tmp/{encoders_filename}"
    joblib.dump(label_encoders, local_encoders_path)
    
    blob_encoders = bucket.blob(f"models/{encoders_filename}")
    blob_encoders.upload_from_filename(local_encoders_path)
    
    # Save metrics
    metrics = {
        "timestamp": timestamp,
        "accuracy": accuracy,
        "classification_report": report,
        "model_path": f"gs://{BUCKET_NAME}/models/{model_filename}",
        "encoders_path": f"gs://{BUCKET_NAME}/models/{encoders_filename}",
        "feature_columns": feature_columns,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "class_distribution": y.value_counts().to_dict(),
        "feature_importance": dict(zip(feature_columns, model.feature_importances_))
    }
    
    metrics_filename = f"metrics_{timestamp}.json"
    local_metrics_path = f"/tmp/{metrics_filename}"
    with open(local_metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    blob_metrics = bucket.blob(f"metrics/{metrics_filename}")
    blob_metrics.upload_from_filename(local_metrics_path)
    
    # Clean up local files
    os.remove(local_model_path)
    os.remove(local_encoders_path)
    os.remove(local_metrics_path)
    
    logging.info("✅ Model and metrics saved successfully!")
    
    # Return metrics for next task
    return {
        "accuracy": accuracy,
        "model_path": f"gs://{BUCKET_NAME}/models/{model_filename}",
        "timestamp": timestamp,
        "feature_importance": dict(zip(feature_columns, model.feature_importances_))
    }

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

# Task 3: Notification and Logging
def log_completion(**context):
    """Log comprehensive completion message with metrics"""
    
    # Get metrics from previous task
    metrics = context['task_instance'].xcom_pull(task_ids='train_model')
    
    completion_message = f"""
    🎉 =================================
    🚀 ML PIPELINE COMPLETED SUCCESSFULLY! 
    🎉 =================================
    
    📊 RESULTS SUMMARY:
    ==================
    🎯 Model Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)
    🕒 Execution Time: {metrics['timestamp']}
    📁 Model Location: {metrics['model_path']}
    
    🧠 MODEL DETAILS:
    ================
    📈 Algorithm: Random Forest Classifier
    🎯 Prediction Target: London Bike Trip Duration (short/medium/long)
    📊 Features Used:
       • Hour of day (start_hour)
       • Day of week (day_of_week) 
       • Month (month)
       • Start station ID (start_station_id)
       • End station ID (end_station_id)
       • Same station return (same_station)
    
    🏆 FEATURE IMPORTANCE:
    =====================
    """
    
    # Add feature importance to message
    for feature, importance in metrics['feature_importance'].items():
        completion_message += f"    • {feature}: {importance:.4f}\n"
    
    completion_message += f"""
    
    💾 ARTIFACTS SAVED:
    ==================
    ✅ Trained model: gs://{BUCKET_NAME}/models/
    ✅ Label encoders: gs://{BUCKET_NAME}/models/
    ✅ Evaluation metrics: gs://{BUCKET_NAME}/metrics/
    ✅ Training data: {PROJECT_ID}.{DATASET_ID}.{TABLE_ID}
    
    🚀 PIPELINE STATUS: ✅ SUCCESS
    ============================
    All components executed successfully:
    ✅ Data extraction from BigQuery
    ✅ Feature engineering and preprocessing  
    ✅ Model training and evaluation
    ✅ Artifact persistence to Cloud Storage
    ✅ Comprehensive logging and monitoring
    
    🎯 Ready for production deployment!
    """
    
    # Log to Cloud Logging
    logging.info(completion_message)
    
    # Also print to stdout for additional visibility
    print(completion_message)
    print("="*50)
    print("🏆 ML PIPELINE EXECUTION COMPLETE! 🏆")
    print("="*50)
    
    return completion_message

notification_task = PythonOperator(
    task_id='log_completion',
    python_callable=log_completion,
    dag=dag,
)

# Set task dependencies
create_dataset_task >> extract_data_task >> train_model_task >> notification_task