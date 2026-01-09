# DataPilot.chat 🚀

**Reimagining Data Exploration, Cleaning & Modeling with 0 Code**

DataPilot is a revolutionary no-code platform that transforms raw datasets into trained ML models seamlessly. From data upload to model deployment, everything happens in one intuitive interface powered by AI.

![DataPilot Banner](static/logo-no-background.png)

---

## ⚠️ IMPORTANT LEGAL NOTICE

**🔒 PRIVATE PROPERTY - UNAUTHORIZED ACCESS PROHIBITED**

This software and all associated files, documentation, and intellectual property are the exclusive property of the copyright holder. 

**STRICTLY PROHIBITED:**
- ❌ Unauthorized copying, distribution, or reproduction
- ❌ Reverse engineering or decompilation
- ❌ Modification, adaptation, or derivative works
- ❌ Commercial use without explicit written permission
- ❌ Redistribution in any form (source code, binaries, documentation)

**LEGAL CONSEQUENCES:**
Violation of these terms may result in civil and criminal penalties under applicable copyright and intellectual property laws. Unauthorized use will be prosecuted to the full extent of the law.

**For licensing inquiries, contact:** [Insert Contact Information]

---

## 🌟 Features Overview

### 📊 **Universal Data Connectivity**
- **File Upload**: CSV, Excel, Parquet files with instant preview
- **Database Integration**: MySQL, PostgreSQL, MongoDB, Snowflake, AWS Redshift, Azure SQL, GCP BigQuery
- **Real-time Preview**: Instant data visualization upon upload

### 🧹 **AI-Powered Data Cleaning**
- **Natural Language Processing**: Clean data using plain English commands
- **Smart Suggestions**: AI-driven recommendations for data quality improvements
- **Advanced Operations**: Missing value imputation, outlier detection, feature engineering
- **Custom Transformations**: Text preprocessing, scaling, encoding

### 📈 **Interactive Data Analysis**
- **Comprehensive Statistics**: Descriptive stats, correlation matrices, distribution charts
- **Data Quality Assessment**: Missing values, duplicates, data type analysis
- **Visual Insights**: Automated chart generation and data profiling

### 🤖 **Automated Machine Learning**
- **Model Recommendations**: AI suggests best algorithms for your data
- **Hyperparameter Tuning**: GridSearch, RandomSearch, Optuna optimization
- **Multiple Algorithms**: Random Forest, XGBoost, LightGBM, CatBoost, SVM
- **Cross-Validation**: Built-in model validation and performance metrics

### 📝 **Jupyter Notebook Integration**
- **Code Generation**: Every action generates corresponding Python code
- **Downloadable Notebooks**: Export complete analysis as .ipynb files
- **Reproducible Research**: Full audit trail of data transformations

---

## 📸 Application Screenshots

### 1. Upload & Database Connection
![Upload and Connect to Database](DataPilot.chat%20Screenshots/1_Upload_Connect_DB.png)
*Upload CSV/Excel files or connect to various databases including MySQL, PostgreSQL, MongoDB, and cloud databases*

### 2. Data Preview
![Data Preview](DataPilot.chat%20Screenshots/2_Preview.png)
*Instant preview of uploaded data with automatic type detection and basic statistics*

### 3. Natural Language Data Cleaning
![Clean Data using Human Language](DataPilot.chat%20Screenshots/3_Clean_Data_using_Human_Language.png)
*Clean your data using simple English commands - no coding required*

### 4. Advanced Natural Language Processing
![Advanced NLP Cleaning](DataPilot.chat%20Screenshots/4_Clean_Data_using_Human_Language2.png)
*Execute complex data transformations through conversational AI interface*

### 5. Custom Cleaning Without Code
![Custom Clean Without Code](DataPilot.chat%20Screenshots/5_Custom_Clean_Withour_Code.png)
*Point-and-click interface for advanced data cleaning operations*

### 6. Advanced Data Details & Analytics
![View Advanced Data Details](DataPilot.chat%20Screenshots/6_View_Advanced_Data_Details.png)
*Comprehensive data analysis with statistical summaries, correlation matrices, and data quality metrics*

### 7. Detailed Data Insights
![Advanced Data Details 2](DataPilot.chat%20Screenshots/7_View_Advanced_Data_Details2.png)
*Deep dive into data characteristics with advanced visualizations and insights*

### 8. Column Selection for Modeling
![Column Selection for Model](DataPilot.chat%20Screenshots/8_Colomn_Selection_for_Model.png)
*Intelligent feature selection interface for machine learning model preparation*

### 9. AI Model Recommendations
![Get Model Recommendation](DataPilot.chat%20Screenshots/9_Get_Model_Recommedation.png)
*AI-powered model suggestions based on your data characteristics and problem type*

### 10. Hyperparameter Tuning
![Hyperparameter Tuning](DataPilot.chat%20Screenshots/10_Hyperparameter_Tuning.png)
*Advanced hyperparameter optimization with GridSearch, RandomSearch, and custom parameter spaces*

### 11. Model Training & Accuracy Metrics
![Train Save Model View Accuracy](DataPilot.chat%20Screenshots/11_Train_Save_Model_View_Accuracy.png)
*Train models with real-time accuracy metrics, cross-validation scores, and performance evaluation*

### 12. Downloadable Jupyter Notebooks
![Downloadable Notebook For Flexibility](DataPilot.chat%20Screenshots/12_Downloadable_Noteook_For_Flexibility.png)
*Export complete analysis as Jupyter notebooks for reproducibility and further customization*

---

## 🚀 Quick Start Guide

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation
```bash
git clone [repository-url]
cd datapilot
pip install -r requirements.txt
```

### Configuration
Create necessary directories:
```bash
mkdir -p tmp/uploads tmp/saved_models tmp/notebooks tmp/sessions
```

Set environment variables:
```bash
export TOGETHER_API_KEY="your_together_ai_api_key"
```

### Run the Application
```bash
python app.py
```

Navigate to `http://localhost:5000` to start using DataPilot.

---

## 📁 Project Architecture

```
datapilot/
├── app.py                      # Main Flask application entry point
├── core.py                     # Core imports and utilities
├── config.py                   # Configuration settings and API keys
├── notebook_logger.py          # Jupyter notebook generation logic
│
├── routes/                     # API endpoints and route handlers
│   ├── upload.py              # File upload and database connection handling
│   ├── cleaning.py            # Data cleaning operations and rollback
│   ├── custom_clean.py        # Advanced cleaning features and API
│   ├── modeling.py            # ML model training and evaluation
│   ├── visualize.py           # Data visualization and chart generation
│   ├── data_details.py        # Comprehensive data analysis endpoints
│   ├── describe.py            # Schema description and data profiling
│   ├── data_access.py         # Data access and retrieval utilities
│   └── db_connect.py          # Database connection management
│
├── services/                   # Business logic and core services
│   ├── session_manager.py     # Session handling and data persistence
│   ├── data_loader.py         # Data loading from files and databases
│   ├── db_loader.py           # Database-specific loading utilities
│   ├── schema_generator.py    # Automatic schema generation
│   ├── kernel_manager.py      # Jupyter kernel management
│   ├── model_trainer.py       # ML model training orchestration
│   ├── model_loader.py        # Model loading and persistence
│   ├── automl_recommender.py  # AutoML recommendations engine
│   ├── explainers.py          # Model explainability features
│   └── save_nb.py             # Notebook generation and export
│
├── templates/                  # HTML templates and UI
│   ├── uploads.html           # File upload interface
│   ├── cleaning.html          # Data cleaning dashboard
│   ├── custom_clean.html      # Advanced cleaning interface
│   ├── modeldev.html          # Model development workspace
│   ├── data_details.html      # Data analysis dashboard
│   ├── describe.html          # Data description interface
│   └── main.html              # Main application layout
│
├── static/                     # Static assets and styling
│   ├── style.css              # Application styling
│   ├── logo-no-background.png # Application logo
│   └── [visualization files]  # Generated charts and plots
│
├── utils/                      # Utility functions
│   ├── sanitizer.py           # Data sanitization and validation
│   ├── ai_utils.py            # AI integration utilities
│   └── logging.py             # Application logging
│
├── DataPilot.chat Screenshots/ # Application screenshots
├── notebooks/                  # Generated Jupyter notebooks
├── uploads/                    # Temporary file storage
├── UPLOAD_FOLDER/             # File upload directory
└── tmp/                       # Temporary files and sessions
    ├── uploads/               # Uploaded files
    ├── saved_models/          # Trained ML models
    ├── notebooks/             # Generated notebooks
    └── sessions/              # User session data
```

---

## 🔧 API Endpoints Reference

### Data Upload & Connection
- `POST /upload-csv/<file_id>` - Upload CSV/Excel/Parquet files
- `POST /connect-db/<file_id>` - Connect to databases
- `GET /cleaning/<file_id>` - Access cleaning interface
- `POST /rollback/<file_id>` - Rollback data changes

### Natural Language Data Cleaning
- `POST /execute-query/<file_id>` - Execute natural language queries
- `POST /api/clean/<file_id>/missing` - Handle missing values
- `POST /api/clean/<file_id>/outliers` - Detect and handle outliers
- `POST /api/clean/<file_id>/duplicates` - Remove duplicates
- `POST /api/clean/<file_id>/transform` - Apply data transformations

### Machine Learning & Modeling
- `GET /modeling/modeldev/<file_id>` - Access model development interface
- `GET /modeling/modeldev/<file_id>/columns` - Get dataset columns and types
- `POST /modeling/modeldev/<file_id>/recommend` - Get model recommendations
- `POST /modeling/modeldev/<file_id>/automl-recommend` - AutoML suggestions
- `POST /modeling/modeldev/<file_id>/train` - Train ML models

### Data Analysis & Insights
- `GET /data_details/<file_id>/overview` - Dataset overview and statistics
- `GET /data_details/<file_id>/correlation_matrix` - Correlation analysis
- `GET /data_details/<file_id>/ai_suggestions` - AI cleanup suggestions
- `GET /describe/<file_id>` - Data schema and profiling

### Visualization & Export
- `POST /visualize/<file_id>` - Generate data visualizations
- `GET /download-notebook/<file_id>` - Download Jupyter notebook
- `POST /save-notebook/<file_id>` - Save analysis as notebook

---

## 🎯 Usage Examples

### 1. Upload and Preview Data
```python
# Upload CSV file
curl -X POST -F "file=@dataset.csv" http://localhost:5000/upload-csv/session123

# Preview uploaded data
curl -X GET http://localhost:5000/cleaning/session123
```

### 2. Natural Language Data Cleaning
```python
# Clean using natural language
curl -X POST -H "Content-Type: application/json" \
  -d '{"query": "remove missing values from Age column using mean strategy"}' \
  http://localhost:5000/execute-query/session123

# Handle outliers
curl -X POST -H "Content-Type: application/json" \
  -d '{"query": "detect and remove outliers from Salary column using IQR method"}' \
  http://localhost:5000/execute-query/session123
```

### 3. Advanced Data Analysis
```python
# Get comprehensive data overview
curl -X GET http://localhost:5000/data_details/session123/overview

# Generate correlation matrix
curl -X GET http://localhost:5000/data_details/session123/correlation_matrix

# Get AI suggestions for data cleaning
curl -X GET http://localhost:5000/data_details/session123/ai_suggestions
```

### 4. Machine Learning Model Training
```python
# Get model recommendations
curl -X POST -H "Content-Type: application/json" \
  -d '{"objective": "classification"}' \
  http://localhost:5000/modeling/modeldev/session123/recommend

# Train selected model with hyperparameter tuning
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "model": "RandomForestClassifier",
    "target": "target_column",
    "features": ["feature1", "feature2", "feature3"],
    "objective": "classification",
    "enable_tuning": true,
    "tuning_method": "gridsearch",
    "cv": 5,
    "optimize_metric": "accuracy"
  }' \
  http://localhost:5000/modeling/modeldev/session123/train
```

### 5. AutoML Recommendations
```python
# Get AutoML model suggestions
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "target": "target_column",
    "features": ["feature1", "feature2", "feature3"],
    "objective": "classification"
  }' \
  http://localhost:5000/modeling/modeldev/session123/automl-recommend
```

---

## 🔌 Database Connectivity

### Supported Database Types
- **SQL Databases**: MySQL, PostgreSQL, SQL Server
- **NoSQL Databases**: MongoDB
- **Cloud Databases**: 
  - AWS Redshift
  - Azure SQL Database
  - Google Cloud BigQuery
  - Snowflake Data Warehouse

### Database Connection Examples

#### MySQL Connection
```json
{
  "dbType": "sql",
  "host": "localhost",
  "port": "3306",
  "username": "mysql_user",
  "password": "mysql_password",
  "database": "sales_db",
  "table": "customers"
}
```

#### PostgreSQL Connection
```json
{
  "dbType": "sql",
  "host": "localhost",
  "port": "5432",
  "username": "postgres_user",
  "password": "postgres_password",
  "database": "analytics_db",
  "table": "user_behavior"
}
```

#### MongoDB Connection
```json
{
  "dbType": "nosql",
  "host": "localhost",
  "port": "27017",
  "username": "mongo_user",
  "password": "mongo_password",
  "database": "ecommerce",
  "table": "products"
}
```

#### Snowflake Connection
```json
{
  "dbType": "snowflake",
  "host": "account.snowflakecomputing.com",
  "port": "443",
  "username": "snowflake_user",
  "password": "snowflake_password",
  "database": "WAREHOUSE_DB",
  "table": "SALES_DATA"
}
```

---

## 🤖 AI-Powered Features

### Natural Language Processing Commands
DataPilot understands and executes commands like:

**Data Cleaning:**
- "Remove missing values from Age column using mean imputation"
- "Drop duplicate rows based on email column"
- "Fill null values in Salary column with median"
- "Remove outliers from Price column using IQR method"

**Data Transformation:**
- "Convert Date column to datetime format"
- "Create dummy variables for Category column"
- "Scale numerical features using StandardScaler"
- "Apply log transformation to Revenue column"

**Data Analysis:**
- "Show correlation between Age and Income"
- "Create histogram of salary distribution"
- "Generate boxplot for Price by Category"
- "Calculate summary statistics for numerical columns"

### AutoML Capabilities
- **Automatic Algorithm Selection**: Analyzes data characteristics to suggest optimal algorithms
- **Hyperparameter Optimization**: Automated tuning using GridSearch, RandomSearch, or Bayesian optimization
- **Feature Engineering**: Automatic feature selection and engineering suggestions
- **Model Comparison**: Comprehensive comparison of multiple algorithms with performance metrics

---

## 📊 Supported File Formats & Data Types

### File Formats
- **CSV** (.csv) - Comma-separated values
- **Excel** (.xlsx, .xls) - Microsoft Excel files
- **Parquet** (.parquet) - Columnar storage format

### Data Types Supported
- **Numerical**: Integer, Float, Decimal
- **Categorical**: String, Object, Category
- **Temporal**: Date, DateTime, Timestamp
- **Boolean**: True/False values
- **Text**: Free-form text data

---

## 🛠️ Technical Dependencies

### Core Framework
```
Flask==2.3.3              # Web framework
Flask-CORS==4.0.0          # Cross-origin resource sharing
```

### Data Processing
```
pandas==2.0.3              # Data manipulation and analysis
numpy==1.24.3              # Numerical computing
polars==0.18.15            # Fast DataFrame library
pyarrow==12.0.1            # Columnar in-memory analytics
```

### Machine Learning
```
scikit-learn==1.3.0        # Machine learning library
xgboost==1.7.6             # Gradient boosting framework
lightgbm==4.0.0            # Gradient boosting framework
catboost==1.2              # Gradient boosting library
lazypredict==0.2.12        # Automated model comparison
```

### Visualization
```
matplotlib==3.7.2          # Plotting library
seaborn==0.12.2            # Statistical data visualization
plotly==5.15.0             # Interactive visualizations
```

### AI Integration
```
together==0.2.7            # Together AI API client
openai==0.27.8             # OpenAI API integration
```

### Database Connectors
```
sqlalchemy==2.0.19         # SQL toolkit and ORM
pymongo==4.4.1             # MongoDB driver
snowflake-connector-python==3.1.0  # Snowflake connector
google-cloud-bigquery==3.11.4      # Google BigQuery client
```

### Jupyter Integration
```
jupyter==1.0.0             # Jupyter notebook
jupyter-client==8.3.0      # Jupyter kernel client
ipykernel==6.25.0          # IPython kernel for Jupyter
```

---

## 🔒 Security & Privacy Features

### Data Security
- **File Sanitization**: Automatic filename sanitization and validation
- **SQL Injection Protection**: Parameterized queries and input validation
- **Session Management**: Secure session handling with encrypted storage
- **Data Validation**: Comprehensive input validation and type checking

### Privacy Protection
- **Local Processing**: All data processing happens locally on your machine
- **No Data Transmission**: Raw data never leaves your environment
- **Temporary Storage**: Automatic cleanup of temporary files and sessions
- **Secure API Keys**: Environment variable-based API key management

### Access Control
- **Session Isolation**: Each user session is completely isolated
- **File Access Control**: Restricted file system access
- **API Rate Limiting**: Built-in rate limiting for API endpoints

---

## 📈 Performance Optimizations

### Data Processing
- **Parquet Storage**: Efficient columnar data serialization
- **Memory Management**: Optimized DataFrame operations with lazy loading
- **Streaming Processing**: Handle large datasets without memory overflow
- **Caching**: Intelligent session-based data caching

### Machine Learning
- **Parallel Processing**: Multi-core utilization for model training
- **Early Stopping**: Prevent overfitting with early stopping mechanisms
- **Memory-Efficient Algorithms**: Optimized algorithms for large datasets
- **Model Persistence**: Efficient model serialization and loading

### Web Application
- **Asynchronous Processing**: Non-blocking operations for better UX
- **Compressed Responses**: Gzip compression for API responses
- **Static Asset Optimization**: Minified CSS and JavaScript
- **Database Connection Pooling**: Efficient database connection management

---

## 🐛 Troubleshooting Guide

### Common Issues & Solutions

#### 1. Kernel Startup Failed
```bash
# Solution: Reinstall Jupyter components
pip uninstall jupyter jupyter-client ipykernel
pip install jupyter jupyter-client ipykernel

# Restart the application
python app.py
```

#### 2. Database Connection Error
```bash
# Check database credentials and connectivity
ping [database_host]
telnet [database_host] [port]

# Verify database driver installation
pip install pymongo sqlalchemy psycopg2-binary
```

#### 3. Memory Issues with Large Files
```bash
# Increase system memory allocation
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1

# Use data sampling for large datasets
# The application automatically handles this
```

#### 4. API Key Issues
```bash
# Set environment variable
export TOGETHER_API_KEY="your_api_key_here"

# Or update config.py directly
TOGETHER_API_KEY = "your_api_key_here"
```

#### 5. File Upload Errors
```bash
# Check file permissions
chmod 755 uploads/
chmod 755 tmp/

# Verify supported file formats
# Only .csv, .xlsx, .parquet are supported
```

---

## 🚀 Advanced Features

### Custom Model Integration
- **Bring Your Own Model**: Import pre-trained models
- **Custom Algorithms**: Implement custom ML algorithms
- **Model Ensembling**: Combine multiple models for better performance
- **Transfer Learning**: Leverage pre-trained models for your domain

### Advanced Analytics
- **Time Series Analysis**: Forecasting and trend analysis
- **Anomaly Detection**: Identify unusual patterns in data
- **Clustering Analysis**: Unsupervised learning capabilities
- **Dimensionality Reduction**: PCA, t-SNE, UMAP implementations

### Enterprise Features
- **Multi-User Support**: Team collaboration capabilities
- **Audit Logging**: Complete audit trail of all operations
- **Role-Based Access**: Different permission levels
- **API Integration**: RESTful API for external integrations

---

## 📞 Support & Resources

### Documentation
- **User Guide**: Comprehensive user documentation
- **API Reference**: Complete API documentation
- **Video Tutorials**: Step-by-step video guides
- **Best Practices**: Data science best practices guide

### Community & Support
- **GitHub Issues**: Report bugs and request features
- **Community Forum**: Connect with other users
- **Email Support**: Direct technical support
- **Training Sessions**: Live training and workshops

### Contact Information
- **Technical Support**: support@datapilot.chat
- **Sales Inquiries**: sales@datapilot.chat
- **Partnership**: partners@datapilot.chat
- **General**: info@datapilot.chat

---

## 📄 License & Legal

### Copyright Notice
© 2024 DataPilot.chat. All rights reserved.

### Terms of Use
This software is proprietary and confidential. Use is subject to license agreement.

### Privacy Policy
We are committed to protecting your privacy and data security. All processing is done locally.

### Third-Party Licenses
This software includes third-party libraries subject to their respective licenses.

---

## 🙏 Acknowledgments

### Technology Partners
- **Together AI** - Advanced language model integration
- **Scikit-learn** - Machine learning algorithms and tools
- **Pandas** - Data manipulation and analysis
- **Flask** - Web application framework
- **Jupyter** - Interactive notebook environment

### Open Source Libraries
We acknowledge and thank the open source community for the excellent libraries that make DataPilot possible.

---

**DataPilot.chat** - Transforming data science from complex coding to simple conversations. 🚀

*Empowering everyone to become a data scientist, regardless of coding experience.*