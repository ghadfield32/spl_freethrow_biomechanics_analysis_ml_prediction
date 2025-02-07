#!/bin/bash
echo "Starting entrypoint.sh..."
echo "Current working directory: $(pwd)"
ls -l

# Source the conda initialization script and activate the environment
. /opt/conda/etc/profile.d/conda.sh
conda activate data_science_ft_bio_predictions

# Debug: Print the activated environment and Python path
echo "Activated environment: data_science_ft_bio_predictions"
echo "Python location: $(which python)"
echo "Listing files in /workspace/notebooks/freethrow_predictions:"
ls -l /workspace/notebooks/freethrow_predictions

# Optionally, comment out the Streamlit launch so it doesn't run automatically.
# streamlit run notebooks/freethrow_predictions/app.py --server.port=8501 --server.address=0.0.0.0

# Instead, keep the container running interactively:
echo "Environment is ready. You can now run your commands manually (e.g., testing streamlit or MLflow)."
exec /bin/bash
