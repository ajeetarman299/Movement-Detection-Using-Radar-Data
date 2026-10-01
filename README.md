# Movement Detection Using Radar Data

Deep learning classifier that reads Doppler radar signals and detects whether a person is absent, approaching, or moving away, built with recurrent networks (RNN, LSTM, GRU) in PyTorch.

Developed at IIT Kharagpur under the guidance of Prof. Koustav Rudra.

## Results

The GRU was selected as the final model. The project write-up reports these figures for the final tuned GRU:

| Metric | GRU (final tuned model) |
| --- | --- |
| Training accuracy | 98% |
| Validation accuracy | 95% |

The run saved in the notebook does not reproduce the 95% validation figure. Its outputs cover one full training run (100 epochs per model), and the values below are read directly from that log. Each model is restored to the checkpoint with the lowest validation loss before evaluation.

| Model | Validation accuracy at best checkpoint | Training accuracy at that epoch | Peak validation accuracy |
| --- | --- | --- | --- |
| Bidirectional RNN | 67.91% (epoch 40) | 61.79% | 67.91% |
| Bidirectional LSTM | 75.37% (epoch 21) | 69.85% | 75.37% |
| Bidirectional GRU | 88.06% (epoch 72) | 98.51% | 89.55% (epoch 81) |

The GRU clearly outperforms the RNN and LSTM on this data, which is why it was chosen. The averaged ensemble of all three models scores 0.85 test accuracy in the saved run (see Notes on how the notebook prints this value).

## Approach

1. **Data collection and labeling.** Raw Doppler radar data was recorded in a lab for three scenarios and labeled by hand:

   | Class | Folder | Label |
   | --- | --- | --- |
   | No person | `N` | 0 |
   | Person approaching | `T` | 1 |
   | Person moving away | `A` | 2 |

2. **Signal pre-processing.** Signals were visualized and filtered to remove noise before modeling. The notebook starts from these processed files and then:
   - reads one float per line from each sample file, skipping non-numeric lines
   - truncates each sequence to its last 350 samples, or pads shorter ones with the sequence mean
   - normalizes the whole dataset with a z-score (subtract mean, divide by standard deviation)
3. **Splits.** 80% training; the remaining 20% is split 80/20 into validation and test (`random_state=42`).
4. **Models.** Three bidirectional recurrent classifiers share the same backbone settings: 3 layers, 128 hidden units, dropout 0.5, and a fully connected head. The GRU head adds batch normalization.
5. **Training.** Cross-entropy loss, Adam (learning rate 0.001), batch size 32, 100 epochs, `ReduceLROnPlateau` on validation loss (factor 0.1, patience 10), and best-checkpoint saving by validation loss.
6. **Ensemble.** The RNN, LSTM and GRU outputs are averaged and the arg max is taken as the ensemble prediction.

## Tech stack

- Python 3.12
- PyTorch 2.14 (models, training loop, data loaders)
- NumPy 2.5 (signal loading, padding, normalization)
- scikit-learn 1.9 (data splits, accuracy metric)
- SciPy 1.18 (imported by the notebook for `scipy.stats`)
- JupyterLab 4.6 (running the notebook)

## Project structure

```
Movement-Detection-Using-Radar-Data/
|-- RADAR_PROJECT_CODE_RNN_LSTM_GRU.ipynb   Pre-processing, RNN/LSTM/GRU models, training, ensemble
|-- requirements.txt                        Pinned dependencies (Python 3.12)
|-- .gitignore
`-- README.md
```

The radar dataset is not included in the repository.

## How to run

### 1. Get the code

```bash
git clone https://github.com/ajeetarman299/Movement-Detection-Using-Radar-Data.git
cd Movement-Detection-Using-Radar-Data
```

### 2. Install dependencies

With [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Or with pip:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On a machine without an NVIDIA GPU, install the smaller CPU build of PyTorch inside the activated environment before installing the requirements (drop the `uv` prefix when using pip):

```bash
uv pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt
```

### 3. Prepare the data

The notebook expects a zip archive with one sub-folder per class, each holding one plain-text file per sample with one numeric value per line:

```
processed_files_separated/
|-- N/   no person
|-- T/   person approaching
`-- A/   person moving away
```

The notebook was written for Google Colab and uses `/content/...` paths. When running locally, update `zip_file_path` and `extract_dir` in the first pre-processing cell, and the folder paths in `class_dirs`, to point at your copy of the data.

### 4. Run the notebook

```bash
jupyter lab RADAR_PROJECT_CODE_RNN_LSTM_GRU.ipynb
```

Run the cells in order. Training saves the best checkpoint of each model as `<ModelName>_best_model.pth` in the working directory.

## Notes

- **Library compatibility.** The code runs unchanged on the pinned versions. Since PyTorch 2.6, `torch.load` defaults to `weights_only=True`; the notebook only saves and loads plain `state_dict` checkpoints, so this change does not affect it.
- **Ensemble accuracy format.** `accuracy_score` returns a fraction between 0 and 1, but the final cell formats it with a `%` sign, so the saved output reads `0.85%`. The value is 0.85 as a fraction.
- **Training log.** The `Train Loss` value printed each epoch is the loss of the last validation batch, because the variable is reused inside the validation loop. The training and validation accuracy columns are computed correctly.

## Credits

Project work carried out at IIT Kharagpur under the supervision of Prof. Koustav Rudra.
