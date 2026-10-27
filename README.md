# Obsidian — Legal Citation Intelligence

A Streamlit app that classifies the treatment of a cited case from its title and surrounding passage. It uses the trained scikit-learn pipeline stored in `best_legal_citation_pipeline.joblib`.

## Project files

- `app.py` — Streamlit interface and prediction flow.
- `best_legal_citation_pipeline.joblib` — trained TF-IDF and calibrated classifier pipeline.
- `Legal_Citation_Text_Classification.ipynb` — notebook describing the dataset and model development.
- `requirements.txt` — Python dependencies.

The raw dataset is not included in this project folder. The app makes predictions with the saved model and does not load the dataset at runtime.

## Requirements

- Python compatible with the pinned scikit-learn version in `requirements.txt`.
- The dependencies listed in `requirements.txt`.
- `best_legal_citation_pipeline.joblib` in the same folder as `app.py`.

## Install and run

From the project directory, create and activate a virtual environment, then install dependencies and start Streamlit.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL in the terminal when the app starts.

## Using the app

1. Enter a case title, a citation passage, or both.
2. Select **Classify passage**.
3. Review the predicted treatment and the probability distribution across the model's outcome classes.

The app includes a dark/light theme switch. Classification probabilities are model outputs and should be interpreted as support for research, not as a substitute for reviewing the source text and authority.

## Model notes

The app reads the classifier and word n-gram configuration from the saved pipeline, so the displayed model details reflect that artifact. The accompanying notebook describes a legal citation dataset and model-development process. Dataset statistics and evaluation scores are not presented in the app because they should be checked against the exact saved model and its evaluation record before being reported.

## Troubleshooting

- **Model file not found:** place `best_legal_citation_pipeline.joblib` beside `app.py`.
- **Model could not be loaded:** install the dependencies in `requirements.txt` in the active environment. The scikit-learn version is pinned because serialized estimators can depend on their library version.
- **Streamlit command not found:** activate the virtual environment and run `python -m streamlit run app.py`.
- **Virtual environment points to a missing Python installation:** remove and recreate `.venv` using an installed Python interpreter, then reinstall the dependencies.
