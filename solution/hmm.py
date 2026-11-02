"""The "stubborn judge": a three-state (H / A / D) Viterbi decoder over per-frame evidence.

Evidence comes from the classifier as a log-likelihood per frame and state. The decoder picks the
single most likely H/A/D sequence, paying for every change of state at the rate possession
really changes in labelled matches -- which is what stops one frame of noise splitting a
possession in two.
"""

import numpy as np

STATES = np.array(["H", "A", "D"])
N_STATES = 3


def count_transitions(labels: np.ndarray) -> np.ndarray:
    """Per-frame state transition matrix counted from H/A/D/X labels. X frames are skipped."""
    code = {s: i for i, s in enumerate(STATES)}
    counts = np.ones((N_STATES, N_STATES))  # add-one, so no transition is impossible
    for a, b in zip(labels[:-1], labels[1:]):
        if a in code and b in code:
            counts[code[a], code[b]] += 1
    return counts / counts.sum(axis=1, keepdims=True)


def viterbi(log_like: np.ndarray, trans: np.ndarray, temper: float) -> np.ndarray:
    """Most likely state index per frame, given (n, 3) log-likelihoods.

    `temper` divides the evidence: neighbouring frames are nearly copies of each other, so
    treating each as independent evidence overcounts it and makes the decoder flip on short runs.
    """
    log_like, log_trans = log_like / temper, np.log(trans)
    n = len(log_like)
    back = np.empty((n, N_STATES), dtype=np.int8)
    score = log_like[0] - np.log(N_STATES)
    for t in range(1, n):
        candidates = score[:, None] + log_trans  # [from, to]
        back[t] = candidates.argmax(axis=0)
        score = candidates.max(axis=0) + log_like[t]

    path = np.empty(n, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
