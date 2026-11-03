# Our solution

Frame-level possession for BAR-ATH from the provided tactical tracking, in three steps:

1. **Features** ([features.py](features.py)) — per frame: each team's distance to the ball, player
   speed, team shape (how high and how spread out each team is, and which way it is moving), where
   the ball was last seen relative to the lines, and each team's distance to the ball when it was
   last and next seen. Plus 1 s, 5 s and 10 s rolling means.
2. **Classifier** — a gradient-boosted tree model gives P(H), P(A), P(D) per frame. It is trained
   **only on other matches**: PFF FC's 64 World Cup 2022 matches ([pff_games.py](pff_games.py)) and
   Metrica's two public sample games ([sample_games.py](sample_games.py)), all labelled by code from
   their event data. Every match is also used with home and away swapped, so the model has no team
   bias. Metrica's games are additionally used with the ball hidden on 40% of live play, as their ball
   is tracked far more often than the competition tracker's.
3. **Viterbi decoding** ([hmm.py](hmm.py)) — turns the per-frame probabilities into whole
   possessions, charging for every change of state at the rate possession changes in the training
   matches. The one free setting (how much to trust neighbouring frames) is chosen on 11 PFF matches
   held out from training.

Nothing is fitted or chosen on the competition match's ground truth. The public 5 minutes were
used only to check that the pipeline runs correctly.

## Reproduce

Python 3.13.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r solution/requirements.txt   # .venv/bin/python on macOS/Linux
git clone --depth 1 https://github.com/metrica-sports/sample-data.git ../external/metrica-sample-data
```

Request the PFF FC 2022 World Cup dataset (free, via the form at
<https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset>) and unpack it to
`../external/pff-wc2022/`.

Point the scripts at the data:

| variable | what | default |
|---|---|---|
| `MTAH_DATA_DIR` | the competition bundle (folder containing `tactical/`) | `./data` |
| `METRICA_SAMPLE_DIR` | `data/` folder of the sample-data clone | `../external/metrica-sample-data/data` |
| `PFF_DIR` | PFF folder containing `Tracking Data/`, `Event Data/`, `Metadata/` | `../external/pff-wc2022/FIFA World Cup 2022` |
| `PFF_CACHE_DIR` | where converted PFF matches are cached | `../external/pff-cache` |

```bash
python solution/pff_games.py         # once, ~15 min with 6 processes: converts PFF to 25 fps
python solution/run.py               # ~10-17 min, writes evaluation/results/hmm.csv
python evaluation/validate_submission.py evaluation/results/hmm.csv
```

The run is deterministic: two runs produce byte-identical files.

## External data

| dataset | source | labels |
|---|---|---|
| PFF FC 2022 World Cup, 64 matches | PFF FC, free on request (link above) | PFF's event data; possession labels derived from it by code |
| Metrica Sports sample games 1 & 2 | <https://github.com/metrica-sports/sample-data> | Metrica's event data; possession labels derived from it by code |

No data about the competition match other than the provided bundle is used.
