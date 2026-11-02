# Our solution

Frame-level possession for BAR-ATH from the provided tactical tracking, in three steps:

1. **Features** ([features.py](features.py)) — per frame: each team's distance to the ball, player
   speed, team shape (how high and how spread out each team is, and which way it is moving), and
   where the ball was last seen relative to the lines. Plus 1 s and 5 s rolling means.
2. **Classifier** — a gradient-boosted tree model gives P(H), P(A), P(D) per frame. It is trained
   **only on Metrica's public sample games 1 and 2**, labelled mechanically from their event data
   ([sample_games.py](sample_games.py)). Each game is used twice, once with the ball hidden on 40% of
   live play to match how often the competition tracker loses it, and with home/away swapped so the
   model has no team bias.
3. **Viterbi decoding** ([hmm.py](hmm.py)) — turns the per-frame probabilities into whole
   possessions, charging for every change of state at the rate possession changes in the sample
   games. The one free setting (how much to trust neighbouring frames) is picked by training on one
   sample game and scoring the other.

Nothing is fitted or chosen on the competition match's ground truth. The public 5 minutes were
used only to check that the pipeline runs correctly.

## Reproduce

Python 3.13.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r solution/requirements.txt   # .venv/bin/python on macOS/Linux
git clone --depth 1 https://github.com/metrica-sports/sample-data.git ../external/metrica-sample-data
```

Point the scripts at the data:

| variable | what | default |
|---|---|---|
| `MTAH_DATA_DIR` | the competition bundle (folder containing `tactical/`) | `./data` |
| `METRICA_SAMPLE_DIR` | `data/` folder of the sample-data clone | `../external/metrica-sample-data/data` |

```bash
python solution/run.py                                  # ~3 min, writes evaluation/results/hmm.csv
python evaluation/validate_submission.py evaluation/results/hmm.csv
```

The run is deterministic: two runs produce byte-identical files.

## External data

| dataset | source | labels |
|---|---|---|
| Metrica Sports sample games 1 & 2 | <https://github.com/metrica-sports/sample-data> | Metrica's event data; possession labels derived from it by code |

No data about the competition match other than the provided bundle is used.
