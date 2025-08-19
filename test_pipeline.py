"""
Local testing script for ML Pipeline components
Run this to test your pipeline logic before deploying to Composer
"""

import pandas as pd
from google.cloud import bigquery
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
PROJECT_ID = "flawless-haven-468319-k2"
DATASET_ID = "ml_dataset"
TABLE_ID = "bike_features"

def test_bigquery_connection():
    """Test BigQuery connection and data access"""
    try:
        client = bigquery.Client(project=PROJECT_ID)
        
        # Test query - just count rows
        test_query = """
        SELECT COUNT(*) as total_rows 
        FROM `bigquery-public-data.london_bicycles.cycle_hire` 
        LIMIT 1
        """
        
        logging.info("🔄 Testing BigQuery connection...")
        result = client.query(test_query).to_dataframe()
        logging.info(f"✅ BigQuery connection successful! Total rows: {result['total_rows'].iloc[0]:,}")
        return True
        
    except Exception as e:
        logging.error(f"❌ BigQuery connection failed: {e}")
        return False

def test_data_extraction():
    """Test the data extraction query"""
    try:
        client = bigquery.Client(project=PROJECT_ID)
        
        # Sample data extraction (smaller limit for testing)
        extract_query = f"""
        SELECT
          rental_id,
          duration,
          CASE 
            WHEN duration <= 900 THEN 'short'
            WHEN duration <= 3600 THEN 'medium'
            ELSE 'long'
          END as duration_category,
          EXTRACT(HOUR FROM start_date) as start_hour,
          EXTRACT(DAYOFWEEK FROM start_date) as day_of_week,
          start_station_id,
          end_station_id
        FROM `bigquery-public-data.london_bicycles.cycle_hire`
        WHERE 
          duration > 60 AND duration < 86400
          AND start_date >= '2022-01-01' 
          AND start_date < '2022-02-01'  -- Just January for testing
          AND start_station_id IS NOT NULL
        LIMIT 1000  -- Small sample for testing
        """
        
        logging.info("🔄 Testing data extraction query...")
        df = client.query(extract_query).to_dataframe()
        
        logging.info(f"✅ Data extraction successful!")
        logging.info(f"📊 Sample size: {len(df)} rows")
        logging.info(f"📊 Columns: {list(df.columns)}")
        logging.info(f"📊 Duration categories: {df['duration_category'].value_counts().to_dict()}")
        
        return df
        
    except Exception as e:
        logging.error(f"❌ Data extraction failed: {e}")
        return None

def test_model_training(df):
    """Test model training logic with sample data"""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import accuracy_score
        
        logging.info("🔄 Testing model training...")
        
        # Prepare features
        feature_columns = ['start_hour', 'day_of_week', 'start_station_id', 'end_station_id']
        X = df[feature_columns].copy()
        y = df['duration_category']
        
        # Simple encoding for testing
        for col in ['start_station_id', 'end_station_id']:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)  # Small for testing
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logging.info(f"✅ Model training successful!")
        logging.info(f"🎯 Test accuracy: {accuracy:.4f}")
        logging.info(f"📊 Training samples: {len(X_train)}")
        logging.info(f"📊 Test samples: {len(X_test)}")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Model training failed: {e}")
        return False

def test_storage_access():
    """Test Cloud Storage access"""
    try:
        from google.cloud import storage
        
        client = storage.Client(project=PROJECT_ID)
        bucket_name = f"{PROJECT_ID}-ml-models"
        
        logging.info("🔄 Testing Cloud Storage access...")
        
        # Try to access bucket
        bucket = client.bucket(bucket_name)
        
        # List existing objects (if any)
        blobs = list(bucket.list_blobs(max_results=5))
        
        logging.info(f"✅ Cloud Storage access successful!")
        logging.info(f"📁 Bucket: gs://{bucket_name}")
        logging.info(f"📄 Objects found: {len(blobs)}")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Cloud Storage access failed: {e}")
        return False

def run_all_tests():
    """Run all pipeline tests"""
    print("="*60)
    print("🧪 ML PIPELINE COMPONENT TESTING")
    print("="*60)
    
    tests_passed = 0
    total_tests = 4
    
    # Test 1: BigQuery connection
    if test_bigquery_connection():
        tests_passed += 1
    
    print("-" * 40)
    
    # Test 2: Data extraction
    df = test_data_extraction()
    if df is not None:
        tests_passed += 1
    
    print("-" * 40)
    
    # Test 3: Model training (only if data extraction worked)
    if df is not None:
        if test_model_training(df):
            tests_passed += 1
    else:
        logging.warning("⚠️ Skipping model training test (no data)")
    
    print("-" * 40)
    
    # Test 4: Storage access
    if test_storage_access():
        tests_passed += 1
    
    print("="*60)
    print(f"🏆 TESTING COMPLETE: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! Your pipeline is ready to deploy.")
    else:
        print("❌ Some tests failed. Check the logs above for details.")
    
    print("="*60)

if __name__ == "__main__":
    run_all_tests()