# Our solution

Frame-level possession for BAR-ATH from the provided tactical tracking, in three steps:

1. **Features** ([features.py](features.py)) — per frame: each team's distance to the ball, player
   speed, team shape (how high and how spread out each team is, and which way it is moving), where
   the ball was last seen relative to the lines, and each team's distance to the ball when it was
   last and next seen, and ball touches (changes of the ball's velocity, credited to the nearest
   team), the direction of each kick from the ball's path before and after it, outside the
   tracker's blur, and each team's distance to two gently de-blurred ball positions that cannot
   over-correct for any plausible blur shape. Plus 1 s, 5 s and 10 s rolling means.
2. **Classifier** — a gradient-boosted tree model gives P(H), P(A), P(D) per frame. It is trained
   **only on other matches**: the 7 matches of the IDSSE open dataset ([idsse_games.py](idsse_games.py)),
   labelled by the official per-frame ball-status and ball-possession flags in their tracking, and
   Metrica's two public sample games ([sample_games.py](sample_games.py)), labelled by code from
   their event data. Their ball is smoothed to look like the competition tracker's
   ([calibrate_ball.py](calibrate_ball.py) sets how much, from aggregates of the competition
   input), and every match is used as is and with the ball hidden as often as the competition
   tracker loses it, each also with home and away swapped. Frames are weighted by 10 s ÷ the
   length of their possession, because the metric counts every possession equally.
3. **Viterbi decoding** ([hmm.py](hmm.py)) — turns the per-frame probabilities into whole
   possessions, charging for every change of state at the rate possession changes in the training
   matches. The one free setting (how much to trust neighbouring frames) is chosen leave-one-match-out
   over the 9 training matches.

Nothing is fitted or chosen on the competition match's ground truth or video. The public 5 minutes
were used only to check that the pipeline runs correctly.

## Reproduce

Python 3.13.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r solution/requirements.txt   # .venv/bin/python on macOS/Linux
```

### From the committed model (no training, no external data)

```bash
export MTAH_DATA_DIR=/path/to/the/competition/bundle   # the folder containing tactical/
python solution/predict.py           # ~3 min, writes evaluation/results/hmm.csv
python evaluation/validate_submission.py evaluation/results/hmm.csv
```

`solution/model/cooked_model.joblib` is the classifier, class prior, transition matrix and temper
that `run.py` trained on the external matches below. `predict.py` reproduces the submission byte
for byte.

### Retraining from scratch

```bash
git clone --depth 1 https://github.com/metrica-sports/sample-data.git ../external/metrica-sample-data
```

Download the IDSSE open data (7 matches, ~2.6 GB: the `DFL_02_01_matchinformation_*`,
`DFL_03_02_events_raw_*` and `DFL_04_03_positions_raw_observed_*` XML files) from
<https://huggingface.co/datasets/pysport/idsse-data> into `../external/idsse/`.

Point the scripts at the data:

| variable | what | default |
|---|---|---|
| `MTAH_DATA_DIR` | the competition bundle (folder containing `tactical/`) | `./data` |
| `METRICA_SAMPLE_DIR` | `data/` folder of the sample-data clone | `../external/metrica-sample-data/data` |
| `IDSSE_DIR` | folder with the IDSSE XML files | `../external/idsse` |
| `IDSSE_CACHE_DIR` | where converted IDSSE matches are cached | `../external/idsse-cache` |

```bash
python solution/idsse_games.py       # once, ~30 s: converts IDSSE to a 25 fps cache
python solution/run.py               # ~45 min: trains, saves solution/model/, writes evaluation/results/hmm.csv
python solution/calibrate_ball.py    # optional, ~2 min: re-derives the ball smoothing in run.py
python evaluation/validate_submission.py evaluation/results/hmm.csv
```

The run is deterministic: two runs produce byte-identical files and the same model.

## External data

| dataset | source | licence | labels |
|---|---|---|---|
| IDSSE: 7 Bundesliga / 2. Bundesliga 2022/23 matches (DFL, Sportec Solutions) | Bassek, M. et al. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data* 12, 195. <https://doi.org/10.1038/s41597-025-04505-y>; mirror <https://huggingface.co/datasets/pysport/idsse-data> | CC BY 4.0 | official per-frame `BallStatus` / `BallPossession` flags in the tracking |
| Metrica Sports sample games 1 & 2 | <https://github.com/metrica-sports/sample-data> | public sample data | Metrica's event data; possession labels derived from it by code |

No data about the competition match other than the provided bundle is used.
