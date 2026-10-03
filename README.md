# 🚗 Used Car Price Analysis and Prediction Using Machine Learning

An end-to-end ML project and Streamlit web app that explores used-car pricing, compares a
**Linear Regression** baseline with a **Random Forest** ensemble, and predicts a price in **₹ INR**
for any car you describe - with a comparison against the brand and market average.

## Features

| Area | What you get |
|---|---|
| Data pipeline | Automatic cleaning (duplicates, missing values, string/unit formats, outliers), `Car_Age = 2026 − Year`, `StandardScaler` + `OneHotEncoder`, train/test split |
| EDA | KPI cards, Age vs Price, Mileage vs Price, Brand vs Avg Price, Fuel box plot, Manual vs Automatic comparison, price distribution, correlation heatmap |
| ML | Linear Regression vs Random Forest, MAE / RMSE / R², train-vs-test R², feature importance, actual-vs-predicted, models saved with `joblib` |
| Predictor | Live form (Brand, Model, Year, Mileage, Fuel, Transmission, Engine, Owners) → `₹ XX,XX,XXX` estimate, likely range, comparison with brand and market average |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The first launch trains both models (about 10-30 s) and saves them to `models/`. After that the
app starts instantly; models are retrained automatically if the dataset changes, or on demand
from the **Model Performance & Training** page.

## Project structure

```
Car_Price_Prediction_Project/
├── app.py               # Streamlit UI: sidebar navigation + 4 pages
├── src/
│   ├── data_loader.py   # loading, synthetic fallback, cleaning, feature engineering, preprocessing
│   ├── eda.py           # KPIs and interactive Plotly charts
│   └── ml_models.py     # training, evaluation, feature importance, persistence, inference
├── data/                # put your CSV here (a sample used_cars.csv is included)
├── models/              # trained models are saved here (created automatically)
├── requirements.txt
├── make_zip.py          # bundles everything into Car_Price_Prediction_Project.zip
└── README.md
```

## Data

* **Local CSV:** the app uses the first usable `*.csv` in `data/` (`used_cars.csv` is checked first).
  Column names are matched flexibly (e.g. `brand/make`, `model_year/year`, `milage/km_driven`,
  `fuel_type`, `engine`, `transmission`, `owner`, `price/selling_price`).
* **Unit handling:** prices written with `$` are converted to INR at ₹83 per $1, mileage in miles is
  converted to km, and engine strings such as `3.5L V6` become CC. Adjust `USD_TO_INR` /
  `MILES_TO_KM` in `src/data_loader.py` if needed.
* **Fuel / transmission:** fuels are normalised to Petrol, Diesel, CNG, Electric (plus Hybrid when
  present); any transmission that is not manual (A/T, CVT, DCT, ...) is treated as Automatic.
* **Owner count:** used only if the CSV contains an owner column. If not (as in the bundled
  `used_cars.csv`), it is left out of the model and the form says so.
* **Synthetic fallback:** if no usable CSV exists - or you pick *Synthetic demo data* in the sidebar -
  a deterministic Indian-market dataset (~6,000 cars, 15 brands) is generated from a depreciation model.

> ⚠️ The bundled `used_cars.csv` is a **US-market** dataset. Its prices are converted to INR, so
> predictions reflect US listing prices, not the Indian market. For India-specific estimates, drop
> an Indian dataset into `data/`.

## Modelling notes

* Models are fitted on `log(1 + Price)` (prices are right-skewed) and evaluated in ₹ on a held-out 20% test set.
* The best model by test R² is saved to `models/<source>/best_model.joblib`; the fitted preprocessor is
  saved to `models/<source>/preprocessor.joblib`.
* The prediction range is the 10th-90th percentile across the Random Forest's trees.
* The more complex model does not always win - the app reports whichever scores best on the test set.

## Packaging

```bash
python make_zip.py     # creates Car_Price_Prediction_Project.zip
```

## Troubleshooting

* **`ModuleNotFoundError: src`** - run `streamlit run app.py` from the project root.
* **Model fails to load after upgrading scikit-learn** - the app retrains automatically; or delete `models/`.
* **Slow first start** - the Random Forest is being trained once; later starts load the saved model.
