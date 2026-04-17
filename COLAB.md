Google Colab GPU workflow

If you prefer a one-click notebook workflow, use [colab/Metabolite_GPrediction_Colab.ipynb](colab/Metabolite_GPrediction_Colab.ipynb).

1. In Colab, switch Runtime -> Change runtime type -> GPU.

2. Clone the repository:

```python
!git clone https://github.com/Gtedget/Metabolite_gprediction.git
%cd Metabolite_gprediction
```

3. Install dependencies:

```python
!pip install -r requirements.txt
```

4. Optional: store MLflow runs in Google Drive.

```python
from google.colab import drive
drive.mount('/content/drive')
MLFLOW_URI = 'file:/content/drive/MyDrive/metabolite_mlruns'
```

If you do not want Google Drive, use:

```python
MLFLOW_URI = 'file:/content/mlruns'
```

If your processed dataset is not checked into the repository, keep it in Google Drive and point Colab at that folder:

```python
from pathlib import Path

DATA_DIR = Path('/content/drive/MyDrive/metabolite_data')
TRAIN_CSV = DATA_DIR / 'train.csv'
VAL_CSV = DATA_DIR / 'val.csv'
TEST_CSV = DATA_DIR / 'test.csv'
TRANSFORM_MAP_JSON = DATA_DIR / 'transform_map.json'
ENZYME_MAP_JSON = DATA_DIR / 'enzyme_map.json'
```

The training script will also look for a sibling `processed_metabolism_data.csv` next to `train.csv` when building the tokenizer.

5. Train on GPU with MLflow enabled:

```python
from datetime import datetime
RUN_NAME = datetime.utcnow().strftime('colab_%Y%m%d_%H%M%S')

!python train.py \
  --data {TRAIN_CSV} \
  --val_data {VAL_CSV} \
  --test_data {TEST_CSV} \
  --transform_map {TRANSFORM_MAP_JSON} \
  --enzyme_map {ENZYME_MAP_JSON} \
  --epochs 30 \
  --batch_size 32 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --dropout 0.2 \
  --device cuda \
  --representation selfies \
  --balance_transform_classes \
  --oversample_strategy coarse_transform \
  --oversample_power 0.5 \
  --scheduler plateau \
  --scheduler_patience 3 \
  --scheduler_factor 0.5 \
  --early_stopping_patience 6 \
  --num_workers 2 \
  --amp \
  --output_dir artifacts \
  --run_name "$RUN_NAME" \
  --use_mlflow \
  --mlflow_experiment metabolite-predictor-colab \
  --mlflow_tracking_uri "$MLFLOW_URI"
```

6. Run generation evaluation:

```python
!python evaluate_generation.py \
  --data {TEST_CSV} \
  --model "artifacts/$RUN_NAME/trained_model.best.pt" \
  --metadata "artifacts/$RUN_NAME/trained_model.metadata.json" \
  --top_k 5 \
  --beam_width 5 \
  --device cuda \
  --out "artifacts/$RUN_NAME/generation_eval.json" \
  --use_mlflow \
  --mlflow_experiment metabolite-generation-eval-colab \
  --mlflow_run_name "$RUN_NAME-eval" \
  --mlflow_tracking_uri "$MLFLOW_URI"
```

7. Run interactive inference:

```python
!python inference.py \
  --model "artifacts/$RUN_NAME/trained_model.best.pt" \
  --metadata "artifacts/$RUN_NAME/trained_model.metadata.json" \
  --top_k 5 \
  --beam_width 5 \
  --device cuda
```

Notes

- The repository is set up to train directly from the included train.csv, val.csv, test.csv, transform_map.json, and enzyme_map.json files.
- Training outputs are isolated under `artifacts/<run_name>/` so repeated Colab runs do not overwrite each other.
- MLflow artifacts will include the saved model, best checkpoint, metadata JSON, training log, and generation-evaluation JSON when requested.
- If you prefer MLflow project execution, you can also run:

```python
!mlflow run . -e train -P device=cuda -P representation=selfies -P epochs=20 -P run_name=colab_project_run
```
