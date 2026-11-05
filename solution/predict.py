"""Label the competition match with the committed model, without retraining.

    python solution/predict.py   # ~3 min, writes evaluation/results/hmm.csv

Needs only the competition bundle (MTAH_DATA_DIR). The model in solution/model/ was written by
run.py, which trains it on external matches only; this reproduces run.py's output byte for byte.
"""

import joblib

import run

if __name__ == "__main__":
    saved = joblib.load(run.MODEL_PATH)
    print("wrote {}".format(run.label_competition((saved["model"], saved["prior"]),
                                                  (saved["sg_model"], saved["sg_prior"]),
                                                  saved["trans"], saved["temper"])))
