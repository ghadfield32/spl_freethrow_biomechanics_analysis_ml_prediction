
import streamlit as st
import pandas as pd
import os
from pathlib import Path
import logging

# Import the unified predict+SHAP function and its helpers.
from ml.predict_with_shap_usage import (
    predict_and_shap,
    load_config as shap_load_config,
    load_dataset as shap_load_dataset,
    setup_logging as shap_setup_logging
)

# (Optionally) set a logging level so you can see info in the terminal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


st.sidebar.header("Prediction & SHAP Options")

# Set a default configuration file path.
default_config_path = "../../data/model/preprocessor_config/preprocessor_config.yaml"
config_path = st.sidebar.text_input("Configuration file path", default_config_path)

# Option to use the same processed ML data file (or upload one)
default_data_path = "../../data/processed/final_ml_dataset.csv"
data_path = st.sidebar.text_input("Prediction data (CSV) path", default_data_path)

# Options for SHAP outputs
generate_summary = st.sidebar.checkbox("Generate SHAP Summary Plot", value=True)
generate_dependence = st.sidebar.checkbox("Generate SHAP Dependence Plots", value=True)
generate_force = st.sidebar.checkbox("Generate Force Plot(s) for a trial", value=True)

# If generating force plots, allow the user to enter a trial ID (or select from a list)
trial_id = st.sidebar.text_input("Trial ID (for force plot & feedback)", value="T0125")


if st.sidebar.button("Run Prediction + SHAP Pipeline"):
    try:
        # Load configuration using the helper function.
        config_path_obj = Path(config_path)
        shap_config = shap_load_config(config_path_obj)
        st.sidebar.success(f"Configuration loaded from {config_path_obj}")
        
        # Load the input data (prediction data)
        data_path_obj = Path(data_path)
        df_input = shap_load_dataset(data_path_obj)
        st.write("### Input Data Preview")
        st.dataframe(df_input.head())

        # Determine the save directory from the configuration (or use a default)
        paths_config = shap_config.get("paths", {})
        save_dir = Path(paths_config.get("predictions_output_dir", "preprocessor/predictions")).resolve()

        # You can add additional parameters if you want to pass dynamic values.
        results = predict_and_shap(
            config=shap_config,
            df_input=df_input,
            save_dir=save_dir,
            generate_summary_plot=generate_summary,
            generate_dependence_plots=generate_dependence,
            generate_force_plots_or_feedback_indices=[trial_id] if generate_force else None,
            top_n_features=len(df_input.columns),
            use_mad=False,
            generate_feedback=True,
            index_column="trial_id",  # ensure your data has a 'trial_id' column
            logger=logger
        )
        st.success("Prediction + SHAP pipeline executed successfully.")
        
        # Display the final dataset (with predictions and feedback) as a table.
        final_dataset_path = results.get("final_dataset")
        if final_dataset_path and os.path.exists(final_dataset_path):
            final_df = pd.read_csv(final_dataset_path, index_col=0)
            st.write("### Final Predictions with SHAP Annotations")
            st.dataframe(final_df.head())
            
            # Display individual feedback for the selected trial.
            if trial_id in final_df.index:
                st.write(f"### Feedback for Trial {trial_id}")
                st.json(final_df.loc[trial_id, "specific_feedback"])
            else:
                st.warning(f"Trial ID {trial_id} not found in the final dataset.")
        else:
            st.error("Final dataset not found. Check logs for errors.")
            
        # Display the paths to generated files.
        st.write("### Generated Files")
        st.write(results)
        
        # Optionally, you might display the SHAP summary plot image.
        summary_plot_path = results.get("shap_summary_plot")
        if summary_plot_path and os.path.exists(summary_plot_path):
            st.write("### SHAP Summary Plot")
            st.image(summary_plot_path, use_column_width=True)
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.error(f"Error in pipeline: {e}")



if st.sidebar.button("Show SHAP Dashboard"):
    # Assume results have already been computed above.
    if not results:
        st.warning("Please run the pipeline first.")
    else:
        tab1, tab2, tab3 = st.tabs(["Global SHAP Plots", "Trial Feedback", "Force Plot"])
        
        with tab1:
            st.header("Global SHAP Plots")
            if summary_plot_path and os.path.exists(summary_plot_path):
                st.image(summary_plot_path, caption="SHAP Summary Plot", use_column_width=True)
            else:
                st.write("No summary plot generated.")
            # Optionally, list out the dependence plot images from the shap_dependence_plots folder.
            dep_plots_dir = save_dir / "shap_dependence_plots"
            if dep_plots_dir.exists():
                st.write("Dependence Plots:")
                for plot_file in sorted(dep_plots_dir.glob("*.png")):
                    st.image(str(plot_file), caption=plot_file.name, use_column_width=True)
            else:
                st.write("No dependence plots directory found.")
        
        with tab2:
            st.header("Individual Trial Feedback")
            if trial_id in final_df.index:
                st.write(f"Feedback for trial {trial_id}:")
                st.json(final_df.loc[trial_id, "specific_feedback"])
            else:
                st.write("Trial not found.")
            st.write("### Final Predictions Table")
            st.dataframe(final_df)
        
        with tab3:
            st.header("Force Plot for Selected Trial")
            # The unified function saves force plots in a specific folder.
            force_plots_dir = save_dir / "shap_force_plots"
            trial_force_plot = force_plots_dir / f"shap_force_trial_{trial_id}.png"
            if trial_force_plot.exists():
                st.image(str(trial_force_plot), caption=f"Force Plot for Trial {trial_id}", use_column_width=True)
            else:
                st.write("Force plot for the selected trial not available.")



