import streamlit as st
import pandas as pd
import os
from pathlib import Path
import logging
import streamlit.components.v1 as components

# Import our unified predict+SHAP function and helpers.
from ml.shap.predict_with_shap_usage import (
    predict_and_shap,
    load_config,
    load_dataset,
    setup_logging
)
from ml.shap.shap_visualizer import ShapVisualizer

# Import the animation function.
from animate_calc_bayes_shap import run_shot_meter_animation

# Optionally set logging to see details in your terminal.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define common viewpoints
COMMON_VIEWPOINTS = {
    "side_view_right": {"elev": 0, "azim": 90},
    "side_view_left": {"elev": 0, "azim": -90},
    "top_down": {"elev": 90, "azim": 0},
    "diagonal_view": {"elev": 45, "azim": 45},
    "player_centric": {"elev": 30, "azim": 0},
    "diagonal_player_centric": {"elev": 30, "azim": 45},
    "inverse_player_centric": {"elev": 30, "azim": 180}
}

########################################
# Helper Functions
########################################
def update_force_plot(selected_trial, results, logger):
    """
    Regenerates the interactive SHAP force plot for a given trial and saves it as 'shap_force_plot.html'
    in the appropriate directory.
    """
    # Determine the output directory (assume same as used in predict_and_shap)
    force_plots_dir = Path(results["final_dataset"]).parent / "shap_force_plots"
    force_plots_dir.mkdir(parents=True, exist_ok=True)
    force_plot_file = force_plots_dir / "shap_force_plot.html"
    
    logger.info(f"Updating force plot for trial '{selected_trial}' and saving to {force_plot_file}.")
    
    # Create an instance of ShapVisualizer
    visualizer = ShapVisualizer(logger=logger)
    # Call the force plot function; note that the API might differ
    visualizer.plot_force(
        results["explainer"],
        results["shap_values"],
        results["X_preprocessed"],
        selected_trial,  # or pass trial id or index as expected by plot_force
        force_plot_file,
        debug=True
    )

def run_animation_pipeline(
    trial_id: str,
    selected_metric: str,
    feedback_mode: str,
    viewpoint_name: str,
    config,
    debug: bool = False,
    polar_plot: bool = True,
    bar_plot: bool = True,
    line_plot: bool = True,
    min_max_range_percentile: int = 10,
    update_percentiles: bool = True,
    bayesian_metrics_json_path: str = "data/model/shot_meter_docs/bayesian_metrics_dict.json",
    merged_data_path: str = "data/processed/final_granular_dataset.csv",
    streamlit_app_paths: bool = True,
    show_selected_metric: bool = True,
    elev: int = 30,
    azim: int = 45,
    data_zoom: float = 1.0
):
    """
    Wraps the run_shot_meter_animation call with parameters collected from the UI.
    Returns the HTML output of the animation and the feedback table.
    """
    # Call the animation function to generate the animation HTML and feedback table.
    animation_html, feedback_table = run_shot_meter_animation(
        bayesian_metrics_json_path=bayesian_metrics_json_path,
        merged_data_path=merged_data_path,
        trial_id=trial_id,
        selected_metric=selected_metric,
        feedback_mode=feedback_mode,
        viewpoint_name=viewpoint_name,
        debug=debug,
        polar_plot=polar_plot,
        bar_plot=bar_plot,
        line_plot=line_plot,
        bayesian_range_percentile=min_max_range_percentile,
        calculated_range_percentile=min_max_range_percentile,
        shap_range_percentile=min_max_range_percentile,
        update_percentiles=update_percentiles,
        config=config,
        streamlit_app_paths=streamlit_app_paths,
        save_path=None,
        notebook_mode=False,
        show_selected_metric=show_selected_metric,
        elev=elev,
        azim=azim,
        data_zoom=data_zoom
    )
    feedback_table = pd.DataFrame(feedback_table)
    # Add debugging outputs to verify the types and content of returned objects.
    if debug:
        st.write("run_shot_meter_animation returned:")
        st.write(" - animation_html type: %s", type(animation_html))
        st.write(" - feedback_table type: %s", type(feedback_table))
        if isinstance(feedback_table, pd.DataFrame):
            st.write(" - feedback_table shape: %s", feedback_table.shape)
            st.write(" - feedback_table columns: %s", feedback_table.columns.tolist())
        else:
            st.write(" - feedback_table content (preview): %s", str(feedback_table)[:500])
    
    return animation_html, feedback_table




########################################
# STEP 1: Session State Initialization #
########################################
if "results" not in st.session_state:
    st.session_state["results"] = None
if "final_df" not in st.session_state:
    st.session_state["final_df"] = None

# Initialize camera state for elev and azim if not already present.
if 'camera' not in st.session_state:
    st.session_state.camera = {
        'elev': 30,
        'azim': 45
        # 'data_zoom' is no longer part of camera.
    }
# Initialize data_zoom directly in session_state if not present.
if "data_zoom" not in st.session_state:
    st.session_state.data_zoom = 2.0



        
########################################
# Sidebar: Configuration & Data Paths  #
########################################
st.sidebar.header("Prediction & SHAP Options")

# Determine project_root relative to this file (adjust as needed)
project_root = Path(__file__).resolve().parent.parent.parent
project_root2 = Path(__file__).resolve().parent.parent
project_root3 = Path(__file__).resolve().parent
print(f"project_root: {project_root}")
print(f"project_root2: {project_root2}")
print(f"project_root3: {project_root3}")

# Update default file paths as needed.
default_config_path = project_root / "data" / "model" / "preprocessor_config" / "preprocessor_config_app.yaml"
default_data_path = project_root / "data" / "processed" / "final_ml_dataset.csv"

#load dataset from default and get the unique trial_id
df = pd.read_csv(default_data_path)
trial_ids = df['trial_id'].unique()
# add a drop down for the user to select the trial_id
trial_id = st.sidebar.selectbox("Select Trial ID", trial_ids, index=0)
# trial_id = st.sidebar.text_input("Trial ID", value="T0125")

# Let the user override the configuration and data file locations.
config_path_input = default_config_path
data_path_input = default_data_path

# Options for what SHAP outputs to generate.
generate_summary=True
generate_dependence=True
generate_force=True

########################################
# Sidebar: Animation Parameter Controls
########################################
st.sidebar.header("Animation Parameters")
feedback_mode = st.sidebar.selectbox("Select Feedback Mode", ["calculated", "shap", "bayesian"], index=2)
min_max_range = st.sidebar.slider("Min and Max Range Percentile", min_value=0, max_value=100, value=10)

#pull unique keys from COMMON_VIEWPOINTS
viewpoint_name = st.sidebar.selectbox("Select Viewpoint", list(COMMON_VIEWPOINTS.keys()), index=6)


########################################
# Sidebar: Camera Controls  #
########################################
st.sidebar.header("Camera Controls")
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    if st.button('↩️ Rotate Left'):
        st.session_state.camera['azim'] -= 5
    if st.button('↪️ Rotate Right'):
        st.session_state.camera['azim'] += 5
with col2:
    if st.button('↑ Tilt Up'):
        st.session_state.camera['elev'] += 5
    if st.button('↓ Tilt Down'):
        st.session_state.camera['elev'] -= 5
with col3:
    # Updated zoom buttons: update st.session_state.data_zoom directly.
    if st.button('🔍 Zoom In'):
        st.session_state.data_zoom = min(2.0, st.session_state.data_zoom * 1.1)
    if st.button('🔎 Zoom Out'):
        st.session_state.data_zoom = max(0.5, st.session_state.data_zoom * 0.9)

# Updated slider: uses st.session_state.data_zoom with key 'data_zoom'
st.sidebar.slider(
    'Data Zoom Level', 
    0.5, 3.0,
    value=st.session_state.data_zoom,
    key='data_zoom',
    help="Adjust the visible court area"
)


# Retrain Model Button
# st.sidebar.subheader("Retrain Model")
# if st.button("Retrain Model using Bayesian Optimized Training"):
#     with st.spinner("Retraining the model using Bayesian optimized parameters..."):
#         r, df = run_prediction_pipeline()
#         st.session_state["results"] = r
#         st.session_state["final_df"] = df
#         st.success("Model retrained successfully!")
config_path_obj = Path(config_path_input)
config = load_config(config_path_obj)  # Now an instance of AppConfig
# Model Information Section
# pull from the config path the model_save_base_dir path
model_save_base_dir = Path(config.paths.model_save_base_dir)
tuning_results = model_save_base_dir / "tuning_results.json"
# load in the json tuning_results
results = None
if os.path.exists(tuning_results):
    results = pd.read_json(tuning_results)
else:
    st.warning("No tuning results found. Please retrain the model.")
# Display the model information
st.sidebar.header("Model Information")

# 1. Create a list of the models from the columns (filter out irrelevant ones)
model_columns = [col for col in results.columns if col not in ["path", "model_name", "metric_value"]]

# 2. Add a dropdown in the sidebar for model selection
model_choice = st.sidebar.selectbox("Choose a model to inspect:", model_columns)

# 3. Pull out the evaluation metrics for the chosen model
eval_metrics_dict = results.loc["Evaluation Metrics", model_choice]

# 4. Display the metrics in the sidebar
if isinstance(eval_metrics_dict, dict):
    # Convert dict to DataFrame
    metrics_df = pd.DataFrame.from_dict(eval_metrics_dict, orient="index", columns=["Value"])
    metrics_df.reset_index(inplace=True)
    metrics_df.rename(columns={"index": "Metric"}, inplace=True)

    st.sidebar.markdown(f"### {model_choice} Evaluation Metrics")
    st.sidebar.table(metrics_df)  # Use st.sidebar.table to show the table in the sidebar
else:
    st.sidebar.warning("No evaluation metrics found for the chosen model.")


    
########################################
# STEP 2: Define the Prediction Pipeline Function #
########################################
def run_prediction_pipeline(df_input_override=None):
    # Load configuration into an AppConfig instance (same type as main.py)
    config_path_obj = Path(config_path_input)
    config = load_config(config_path_obj)  # Now an instance of AppConfig
    paths = config.paths  # Direct access to the paths object from AppConfig

    # Debug prints
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Computed project_root: {project_root}")
    logger.info(f"Config path: {config_path_obj.resolve()}")
    logger.info(f"Data path: {data_path_input}")
    logger.info(f"Config paths: {paths}")
    
    # Load input data using the configured paths
    if df_input_override is not None:
        df_input = df_input_override
    else:
        data_path_obj = Path(data_path_input)
        df_input = load_dataset(data_path_obj)
    
    # Determine where to save prediction outputs using configuration values
    save_dir = Path("data/preprocessor/predictions")
    
    # Construct model_save_dir_override from the AppConfig
    model_save_dir_override = Path("data/model")
    
    # Base directory for feature info (if needed) can also use a path from config (or define it similarly)
    base_dir = Path("data/preprocessor/features_info")
    
    predictions_output_path = Path("data/predictions")
    results = predict_and_shap(
        config=config,
        df_input=df_input,
        save_dir=predictions_output_path,
        columns_to_add=['trial_id'],
        generate_summary_plot=True,
        generate_dependence_plots=True,
        generate_force_plots=True,
        force_plot_indices=[0],  # Example: first trial by index
        top_n_features=10,  # Adjust as needed
        use_mad=False,
        logger=logger,
        features_file=(Path(config.paths.data_dir) / config.paths.features_metadata_file).resolve(),
        ordinal_file=Path(f'{base_dir}/ordinal_categoricals.pkl'),
        nominal_file=Path(f'{base_dir}/nominal_categoricals.pkl'),
        numericals_file=Path(f'{base_dir}/numericals.pkl'),
        y_variable_file=Path(f'{base_dir}/y_variable.pkl'),
        model_save_dir_override=Path(config.paths.model_save_base_dir),
        transformers_dir_override=Path(config.paths.transformers_save_base_dir),
        metrics_percentile=10
    )
    
    # Attempt to load the final dataset if it exists
    final_dataset_path = results.get("final_dataset")
    if final_dataset_path and os.path.exists(final_dataset_path):
        final_df = pd.read_csv(final_dataset_path, index_col=0)
    else:
        final_df = None

    return results, final_df

########################################
# STEP 3: Run Pipeline Automatically on Startup
########################################
if st.session_state["results"] is None:
    with st.spinner("Running initial prediction pipeline..."):
        r, df = run_prediction_pipeline()
        st.session_state["results"] = r
        st.session_state["final_df"] = df

########################################
# STEP 4: Allow Re-run with New CSV Data
########################################
# st.sidebar.header("Load or Re-run with New Data")
# uploaded_file = st.sidebar.file_uploader("Choose a CSV", type=["csv"])
# if st.sidebar.button("Predict on new CSV"):
#     if uploaded_file is not None:
#         df_new = pd.read_csv(uploaded_file)
#         with st.spinner("Predicting on new CSV..."):
#             r, df = run_prediction_pipeline(df_input_override=df_new)
#             st.session_state["results"] = r
#             st.session_state["final_df"] = df
#             st.success("Re-run with new data complete!")
#     else:
#         st.warning("Please upload a CSV before re-running.")

########################################
# STEP 5: Main Layout with Tabs for Feedback
########################################
results = st.session_state["results"]
final_df = st.session_state["final_df"]

# ------------------------------------------------------------------
# Create two tabs: Animation Feedback and Global Feedback
# ------------------------------------------------------------------
tab_animation, tab_global, tab_feedback_details = st.tabs([
    "Animation Feedback", "Global Feedback", "Feedback Mode Details"
])


# ------------------------------------------------------------------
# Animation Feedback Tab
# ------------------------------------------------------------------
with tab_animation:
    st.header("Animation Feedback Module")

    # Add the Quick Summary text above the feedback table
    st.markdown("""
    ## Quick Summary
    
    Visualize feedback types linked to key moments in the shooting animation:
    
    **"Early" / "Late" / "Good" Classification**
    - **How it works:** Uses your successful shots to define ideal ranges.
    - **Example:** If your elbow angle is too low, you’ll see *Early*; if within range, *Good*.

    **Holistic Calculated Feedback mode (calculated)**
    - **Only for elbow/shoulder/wrist release/max metrics**
    - **What it does:** Basic calculations to define a baseline for a normal improvement method, based on your body mechanics during made shots.
    - **Why it matters:** Helps us to understand what a basic program would look like for feedback on biomechanics.
    
    **Personalized Adjustments for Individual Metrics (shap)**
    - **What it does:** AI identifies exactly what to tweak (e.g., "Raise elbow by 5°").
    - **Visual Cue:** An arrow ▲ or ▼ appears over the joint needing adjustment.
    - **Why it matters:** Helps you understand how to improve your shot accuracy based on your unique body mechanics on individual body mechanic metrics.
    
    **Optimized Combined Metrics (bayesian)**
    - **What it does:** Uses advanced math to calculate your best possible combined metrics for the best possible accuracy.
    - **Why it matters:** Helps you understand how to improve your shot accuracy based on your unique body mechanics and how to better them collectively.
    """)
    
    # --- Step A: Load the Bayesian metrics JSON to get available metric keys ---
    import json
    bayes_metrics_json_path = "data/model/shot_meter_docs/bayesian_metrics_dict.json"
    try:
        with open(bayes_metrics_json_path, "r") as f:
            bayes_metrics = json.load(f)
        metric_options = list(bayes_metrics.keys())
        # Optional: sort the keys alphabetically for easier navigation
        metric_options.sort()
    except Exception as e:
        st.error(f"Error loading bayesian metrics JSON: {e}")
        metric_options = ["elbow_max_angle"]  # fallback option

    # --- Step B: Create a dropdown for selecting the metric ---
    selected_metric = st.selectbox("Select Metric", metric_options, 
                                   index=metric_options.index("elbow_max_angle") if "elbow_max_angle" in metric_options else 0)
    
    # Run the animation pipeline using sidebar parameters.
    animation_output, feedback_table = run_animation_pipeline(
        trial_id=trial_id,                     # Use the sidebar trial ID
        selected_metric=selected_metric,       # Use the selected metric from the dropdown
        feedback_mode=feedback_mode,           # Taken from sidebar
        viewpoint_name=viewpoint_name,  
        config=load_config(Path(config_path_input)),
        debug=False,
        polar_plot=True,
        bar_plot=True,
        line_plot=True,
        min_max_range_percentile=min_max_range,
        update_percentiles=True,
        streamlit_app_paths=True,
        show_selected_metric=True,
        elev=st.session_state.camera['elev'],
        azim=st.session_state.camera['azim'],
        # Updated: pass the new zoom state
        data_zoom=st.session_state.data_zoom
    )
    
    st.success("Animation generated!")
    
    # Debug prints: Inspect feedback_table before rendering
    # st.write("DEBUG: Type of feedback_table:", type(feedback_table))
    # st.write("DEBUG: Preview of feedback_table (first 5 rows if DataFrame):")
    # if isinstance(feedback_table, pd.DataFrame):
    #     st.write(feedback_table.head())
    # else:
    #     st.write(feedback_table)
    
    # Display the full feedback table in an expander for review.
    if feedback_table is not None:
        with st.expander("Feedback Table Preview"):
            st.table(feedback_table)
            st.write("Data Types:", feedback_table.dtypes)
    else:
        st.info("Feedback table not available yet.")

    
    # Display the animation output.
    if isinstance(animation_output, str) and animation_output.endswith(".html"):
        with open(animation_output, "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=2000)
    else:
        components.html(animation_output, height=2000)

# ------------------------------------------------------------------
# Global Feedback Tab
# ------------------------------------------------------------------
with tab_global:
    st.header("Global Feedback")
    
    # Display SHAP Summary Plot if available.
    summary_plot_path = results.get("shap_summary_plot")
    if summary_plot_path and os.path.exists(summary_plot_path):
        st.image(summary_plot_path, caption="SHAP Summary Plot", use_container_width=True)
    else:
        st.write("No SHAP summary plot available.")
    
    # Display SHAP Dependence Plots (if the folder exists).
    if results.get("final_dataset"):
        dep_dir = Path(results["final_dataset"]).parent / "shap_dependence_plots"
        if dep_dir.exists():
            st.subheader("SHAP Dependence Plots")
            for plot_file in sorted(dep_dir.glob("*.png")):
                st.image(str(plot_file), caption=plot_file.name, use_container_width=True)
        else:
            st.write("No dependence plots available.")
    
    
    # Metric Progression Plot
    if final_df is not None:
        st.subheader("Metric Progression")
        metrics = [
            col for col in final_df.columns
            if any(keyword in col.lower() for keyword in ["ball", "angle", "speed", "velocity"])
        ]
        if metrics:
            metric_choice = st.selectbox("Choose a metric to plot across trials", metrics)
            df_plot = final_df.sort_values("trial_id")
            st.line_chart(df_plot, x="trial_id", y=metric_choice)
        else:
            st.write("No metric columns found for plotting.")


# ------------------------------------------------------------------
# Feedback Mode Details Tab
# ------------------------------------------------------------------
with tab_feedback_details:
    st.header("Feedback Mode Details")
    
    st.markdown("""
    # Integrated Feedback Overview
    
    Using data from your successful shots, our system provides a multi-layered evaluation of your shooting mechanics by combining three primary feedback strategies:
    
    ## 1. Holistic Calculated Feedback (Body Mechanics Baseline)
    
    **Scope & Metrics:**
    
    - **Release Angles:**
        - Extracts angles (e.g., elbow, shoulder, wrist) at the moment of release from successful shots.
        - The optimal value is the mean of these angles.
    - **Max Angles:**
        - Identifies peak joint angles during the shooting motion from made shots.
        - The optimal value is computed as the mean of these peaks.
    
    **Feedback Logic:**
    
    - Compares your current shot’s angles to these established baselines.
    - For example, if your shoulder’s peak is typically 90° but a trial peaks at 85°, the shot is flagged as **Early** (too low).
    
    **Why It Matters:**
    
    It establishes your foundational body mechanics, setting a clear “normal” against which all adjustments are measured.
    
    ---
    
    ## 2. Personalized Adjustments (SHAP)
    
    **Process:**
    
    - **Model-Specific Analysis:**
        - Automatically detects your analysis model (e.g., XGBoost) and selects the appropriate SHAP explainer.
    - **Individual Metric Feedback:**
        - **Positive SHAP Value:** Indicates that increasing a specific metric might improve performance (e.g., “raise wrist by 3°”).
        - **Negative SHAP Value:** Suggests that decreasing the metric could be beneficial (e.g., “lower shoulder rotation by 5°”).
    - **Visual Cues:**
        - Animated indicators (▲ or ▼) overlay the respective joint during key motion phases.
    
    **Example:**
    
    - If your elbow angle is 85° and a SHAP value of –2.1 is computed, the feedback might read:  
      *"Lower elbow bend by 8° (Goal: 77° ±2°)."*
    
    **Why It Matters:**
    
    It tailors the advice to your unique shooting mechanics, providing precise, actionable recommendations.
    
    ---
    
    ## 3. Optimized Combined Metrics (Bayesian)
    
    **Workflow:**
    
    - **Multi-Metric Optimization:**
        - Tests hundreds of combinations of elbow, shoulder, and wrist angles to determine the optimal mix that maximizes shot success probability.
    - **Key Steps:**
        - **Search Space:** Uses normalized ranges derived from your shot data.
        - **Objective Function:** Continuously updates baseline metrics with candidate values and predicts success.
        - **Early Stopping:** Halts if improvements fall below 2% for 5 consecutive iterations.
    - **Output:**
        - **Optimized Values:** The best combination of angles recommended for your body mechanics.
        - **Comparison Table:** Shows your current baseline versus the optimized values and the expected success rate gains.
    
    **Example Output:**
    
    | Metric            | Your Avg | Optimized | Change  |
    |-------------------|----------|-----------|---------|
    | Elbow Release     | 88°      | 92°       | +4°     |
    | Shoulder Max      | 102°     | 98°       | -4°     |
    | Success Rate      | 72%      | 89%       | +17%    |
    
    **Why It Matters:**
    
    By revealing how small combined adjustments can lead to significant performance gains, this method shows you the potential impact of fine-tuning your mechanics holistically.
    
    ---
    
    **Integration into User Interface:**
    
    - **Animation:**
        - **Early/Late Indicators:** Joints highlighted in red with clear boundary markers (e.g., “Elbow: 85° vs. Ideal 88°–92°”).
        - **SHAP Arrows:** Animated pulses (▲/▼) that are synchronized with key phases of the motion.
    - **Tooltips & Badges:**
        - Hover over “Optimized” badges to reveal detailed Bayesian comparison tables.
    - **Dependencies:**
        - The Bayesian feedback is contingent on having a model accuracy above 80% to ensure reliability.
    """, unsafe_allow_html=True)
    
    st.info("Review the above details to understand how each feedback mode works and why each is important for enhancing your shot mechanics.")
    
    

########################################
# End of App
########################################
