# Customer Churn Prediction Project

## Project Overview

This project aims to predict customer churn using machine learning techniques. By analyzing customer data such as demographics, account activity, and service usage, the model identifies customers at high risk of churning, enabling proactive retention strategies. This end-to-end pipeline encompasses data ingestion, preprocessing, model training, and deployment, leveraging Python and various data science libraries.

## Key Features & Benefits

*   **Churn Prediction:** Accurate prediction of customers likely to churn.
*   **Data Pipeline:** Automated end-to-end pipeline for data ingestion, preprocessing, and model training.
*   **Model Deployment:** Instructions to deploy the trained model for real-time predictions.
*   **Data Validation:** Ensures data quality and consistency throughout the pipeline.
*   **Scalability:** Designed to handle large datasets efficiently.
*   **GitHub Actions:** Implemented CI/CD pipeline for automated testing and deployment.

## Prerequisites & Dependencies

Before you begin, ensure you have met the following requirements:

*   **Python:** (>=3.7)
*   **pip:** Python package installer
*   **Virtual Environment (Optional but recommended):** Conda or venv
*   **MongoDB:** For data storage (configure connection details)
*   **MLflow:** For experiment tracking and model management

The required Python libraries are listed in `requirements.txt`.

## Installation & Setup Instructions

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/kazwij/customerchurn.git
    cd customerchurn
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv .venv  #or use conda
    source .venv/bin/activate  # On Linux/macOS
    .venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure MongoDB:**

    *   Ensure MongoDB is installed and running.
    *   Update the connection details in the appropriate configuration files (e.g., within `customerchurn/constant/`) to point to your MongoDB instance.

5.  **Set up MLflow:**

    * Install MLflow `pip install mlflow`
    * You need a MLflow tracking server, default is `mlflow.db`

## Usage Examples & API Documentation

### Running the Training Pipeline

To execute the model training pipeline:

```bash
python main.py
```

This will trigger the entire pipeline, including data ingestion, transformation, training, and model evaluation.  The artifacts like model files and preprocessors will be saved under `final_model/`.

### Pushing data to Database

Use the below script to upload data to MongoDB

```bash
python push_data.py
```

This will fetch the data from `customerchurn_data/Churn_Modelling.csv` and stores into `MongoDB`.

### Example Code Snippets

Example of Data Ingestion component:

```python
from customerchurn.components.data_ingestion import DataIngestion

data_ingestion = DataIngestion(data_ingestion_config) #data_ingestion_config from config_entity
data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
```

## Configuration Options

The project utilizes configuration files to manage settings.  Key configuration options include:

*   **Data Source:** The location of the customer churn dataset (`customerchurn_data/Churn_Modelling.csv`).
*   **MongoDB Connection:** The connection string for accessing the MongoDB database.
*   **Feature Configuration:** Settings related to feature selection and transformation, defined in `data_schema/schema.yaml`.

Adjust these settings by modifying the relevant files in the `customerchurn` directory or through environment variables.

## Contributing Guidelines

We welcome contributions! To contribute:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Implement your changes.
4.  Submit a pull request with a clear description of your changes.

Please adhere to the existing code style and include relevant tests.

## License Information

This project has no specified License. All rights are reserved.

## Acknowledgments

*   We would like to acknowledge the creators and maintainers of the libraries and frameworks used in this project, including scikit-learn, pandas, and MongoDB.
