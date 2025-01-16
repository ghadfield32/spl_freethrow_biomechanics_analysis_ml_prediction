# SPL Biomechanical ML Feedback System

Biomechanical Free Throw SHAP Feedback Application

This application is designed to provide interpretable machine learning feedback for biomechanical data collected during free throw shooting. Using SHAP (SHapley Additive exPlanations) values, the tool not only predicts key metrics from each trial but also offers detailed, individualized feedback, including a "shot meter" for each measured metric.
Table of Contents

    Overview
    Features
    Project Structure
    Setup and Installation
    Usage Instructions
    Pipeline Details
    Dashboard Functionality
    Future Enhancements
    License

Overview

The goal of this project is to give actionable feedback based on free throw biomechanics. The application uses a unified pipeline that processes the input data through a predictive model and generates detailed SHAP outputs, including:

    Global SHAP Summary and Dependence Plots that explain the model's behavior.
    Individual Trial Feedback where each trial receives a detailed feedback score and a "shot meter" that indicates performance on each metric.
    Force Plot Generation for visualizing the contribution of different features in a specific trial.

Features

    Configurable Input: Easily specify the configuration file and CSV dataset paths.
    Prediction & SHAP Pipeline: A one-click solution that loads the configuration, processes data, runs predictions, and generates SHAP explanations.
    Trial-Specific Feedback: Retrieves detailed feedback and individual force plots for selected trial IDs.
    Interactive Dashboard: Offers a sidebar for execution options and a tabbed layout for global and trial-specific visualizations.
    Automated Output Saving: Predictions, feedback, and SHAP visualizations are automatically saved to the designated output directories.

Project Structure

.
├── app.py                  # Main Streamlit application entry point.
├── ml/
│   └── predict_with_shap_usage.py  # Contains the unified function to predict and calculate SHAP values.
├── data/
│   ├── model/
│   │   └── preprocessor_config/
│   │       └── preprocessor_config.yaml   # YAML configuration for preprocessing and model settings.
│   └── processed/
│       └── final_ml_dataset.csv         # Sample or processed ML dataset in CSV format.
└── README.md               # This file.

Setup and Installation

    Clone the Repository:

git clone https://your-repo-url.git
cd your-repo-folder

Install Dependencies:

Ensure you have Python 3.7 or later. Install the required packages using pip:

    pip install -r requirements.txt

    (If you are using streamlit, pandas, and other dependencies not yet included, be sure to add them to your requirements.txt.)

    Directory Structure:

    Make sure the directories referenced in the configuration file and in the code exist (such as data/model/preprocessor_config and data/processed). Adjust the paths in app.py if needed.

Usage Instructions

    Run the Application:

    Launch the Streamlit application:

    streamlit run app.py

    Configure Options:
        Sidebar Inputs:
            Configuration file path: Path to your YAML configuration file (defaults to ../../data/model/preprocessor_config/preprocessor_config.yaml).
            Prediction Data Path: Path to the CSV file containing your processed biomechanical free throw data (defaults to ../../data/processed/final_ml_dataset.csv).
            SHAP Plot Options: Toggle checkboxes to generate the summary, dependence, and force plots.
            Trial ID: Input a specific trial ID to view individual feedback and force plots.

    Run the Pipeline:

    Click on the "Run Prediction + SHAP Pipeline" button. The application will:
        Load the configuration and input data.
        Process the data through the prediction and SHAP pipeline.
        Save and display a preview of the predictions along with feedback.
        Generate and save visual outputs (plots) in the specified directories.

    View the Dashboard:

    After running the pipeline, click on "Show SHAP Dashboard" to access:
        Global SHAP Plots: Overview of SHAP summary and dependence plots.
        Trial Feedback: Detailed feedback and a "shot meter" view of predictions for the selected trial.
        Force Plot: A focused force plot for the trial specified.

Pipeline Details

The core functionality is built around the predict_and_shap function imported from ml.predict_with_shap_usage. This function takes care of:

    Loading the configuration file.
    Reading the input dataset.
    Running predictions with the model.
    Computing SHAP values for explanations.
    Generating outputs such as summary, dependence, and force plots.
    Saving the final dataset which includes predictions and trial-specific feedback.

Additional parameters such as the index_column (set to "trial_id") ensure that each trial is uniquely identified, enabling targeted feedback and visualizations.
Dashboard Functionality

The Streamlit dashboard is divided into three tabs:

    Global SHAP Plots: Displays the overall summary and dependence plots generated from the SHAP analysis.
    Trial Feedback: Shows a detailed prediction and specific feedback for the selected trial, along with a table view of all trial predictions.
    Force Plot: Provides an in-depth force plot for the specified trial, illustrating how each feature contributes to the prediction.

Future Enhancements

    Enhanced Shot Meter: Fine-tuning the “shot meter” to provide more granular feedback on each performance metric.
    Expanded Feedback Module: Integrating additional metrics and personalized coaching tips based on biomechanical insights.
    User Authentication and Session Storage: To retain the state across user sessions and provide a more dynamic user experience.
    API Packaging: Packaging the model and SHAP explanation functions as an API for seamless integration with other applications or real-time feedback systems.

License

This project is licensed under the MIT License. See the LICENSE file for details.
