import sys
from pathlib import Path

# Insert the project root so that absolute imports can work.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import your tasks – use the full package path.
from notebooks.freethrow_predictions.ml.jobs import tuning_job as tuning
from notebooks.freethrow_predictions.ml.jobs import training_job as training
from notebooks.freethrow_predictions.ml.jobs import inference_job as inference

from invoke import Collection

ns = Collection()
ns.add_collection(tuning, name="tuning")
ns.add_collection(training, name="training")
ns.add_collection(inference, name="inference")
ns.configure({'run': {'echo': True}})
