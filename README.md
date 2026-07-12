# Runalyze Tools

A collection of Python utilities which use the Runalyze personal API.

## What is included

### `google-fit-to-runalyze.py`
Syncs walking data from Google Fit to Runalyze to also have these included in the TRIMP calculations.

### `libra-csv-to-runalyze.py`
Reads a Libra weight CSV export and uploads missing weight entries to Runalyze.

### `sync-runalyze.py`
Copies Garmin TRIMP data from a primary Runalyze account to a secondary one.
This way, the Runalyze tools for analyzing TRIMP can be used with the Garmin TRIMP values.

### `perfect-clothing.py` and `perfect-clothing-ui.py`
Uses the data from Runalyze to train an ML model which predicts which outfit should be worn given some conditions such as the temperature.
The GUI for the prediction can be launched with `perfect-clothing-ui.py`.

### `runalyze/`
Runalyze API helper package which provides Python wrappers for some of the API endpoints.


## Setup

1. (Optional) Create and activate a Python virtual environment.

   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

2. Install dependencies.

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Make sure the following credential files are available in the repository root (depending on the tools you want to use):
   - `client_secret.json` for Google OAuth (used by `google-fit-to-runalyze.py`).
     This can be obtained from [console.cloud.google.com](https://console.cloud.google.com/)
   - `runalyze_credentials.json` for the primary Runalyze account.
     This can be created from [runalyze.com/settings/personal-api](https://runalyze.com/settings/personal-api) and is used by all tools.
     Please note that free users can't read data from Runalyze and thus can only use `google-fit-to-runalyze.py`.
   - `runalyze_credentials_secondary.json` for the secondary Runalyze account (the sync target in `sync-runalyze.py`)

> Keep credential files private and do not commit them to version control.

## Usage

### Google Fit to Runalyze

```bash
python google-fit-to-runalyze.py
```

This script will:
- authorize with Google Fit using `client_secret.json`
- read walking and biking distance data
- prompt for confirmation for unusual values
- upload TCX activities for walking to Runalyze using `runalyze_credentials.json`.
  These files only contain the duration at a default speed of 10min/km.

### Libra CSV to Runalyze

```bash
python libra-csv-to-runalyze.py path/to/your/libra-export.csv
```

This script will:
- read the Libra CSV export
- compare weight entries against existing Runalyze body composition data
- upload only missing entries

### Sync secondary Runalyze account

```bash
python sync-runalyze.py
```

This script will:
- read activities from the main Runalyze account
- copy Garmin TRIMP values to the secondary account as time in minutes (using a max HR of 198bpm and an average activity HR of 131bpm this results in the Runalyze TRIMP on the secondary account being exactly the same as the duration in minutes)

### Perfect Clothing

The perfect-clothing tools are meant to be used together:
- the CLI prepares training data from Runalyze and trains the clothing prediction model,
- the UI loads the trained model and lets you make live outfit recommendations.

Run the CLI first to build or update the model from your Runalyze activity data.

```bash
python perfect-clothing.py load_data
python perfect-clothing.py train
```

This trains two models.
The first one uses more parameters, but assigns all data randomly to training and validation sets.
This is very prone to overfitting, as this the augmented activities are very similar to the original activities and thus the model learns these activities specifically.
The second one used less parameters, but uses a `StratifiedGroupKFold` strategy based on the original activity to create training and validation sets.
This way, an activity and it's derived activities are either in the training or validation set, but not both.

If you already have cached Runalyze activity data, you can update it later (instead of downloading everything again):

```bash
python perfect-clothing.py update_data
```

Once the model is trained, launch the UI:

```bash
python perfect-clothing-ui.py
```

Open the browser URL shown in the console and adjust temperature, pace, duration, wind and other inputs to get a prediction.
The upper panel shows the first model (with more parameters) while the lower panel shows the second model (with less parameters).

Note: All project-wide assumptions (thresholds, mappings, clothing lists, and helper functions) are consolidated in `perfect_clothing/assumptions.py`.
To adjust the behavior of the Perfect Clothing tools for your setup, update that single file -- no other source files typically need modification.

The expected data structure is that activities have some clothing as equipment.
The clothing is encoded in `assumptions.SORTED_CLOTHING` by category and then sorted ascending in terms of warmth (i.e. the warmest item comes last).
The categories are dynamic and can be chosen arbitrarily except for `lower_body` and `upper_body`, which must be present.
The temperature feeling is then encoded as tags which are defined in `assumptions.TEMPERATURE_LABEL_MAPPING`.
No tag means the temperature was ok.
`zuHeiss` means that it was too warm but this was not due to clothing (i.e. there no clothing left to lose) while `zuWarmAngezogen` means less clothing would have been possible.
If the temperature was on the edge this currently encoded by the activity note.
If it contains any of the strings in `assumptions.ALMOST_TOO_COLD_WORDS` or `assumptions.ALMOST_TOO_WARM_WORDS`, the current outfit is labeled as fine, but anything slightly warmer (or colder, depending on the words) is labeled as too warm/cold.

## Notes

- The repository currently uses local credential JSON files and config text files for state.
- I haven't tested setting this up from scratch -- if you should encounter any issues please create an issue and I will be happy to help and possibly update this readme.
