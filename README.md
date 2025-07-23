# VibeChecker

VibeChecker is a Python library for measuring prosodic accommodation between two speakers in a conversation. It extracts low‑level audio features, applies configurable accommodation strategies, and produces both turn‑level and sliding‑window summary metrics.

<img width="2000" height="1500" alt="audio" src="https://github.com/user-attachments/assets/f04f3dfb-3191-4c9d-8ef5-ac112c72b16c" />

---

## Table of Contents

1. [Accomodation Types & Strategies](#accomodation-types--strategies)
2. [Generated Metrics & States](#generated-metrics--states)
3. [Audio Features](#audio-features)
4. [Pipeline & CLI Options](#pipeline--cli-options)
5. [Installation](#installation)
6. [File Structure](#file-structure)
7. [Usage](#usage)
8. [Programmatic API](#programmatic-api)
9. [License](#license)

---

## What Is Prosodic Accommodation?

Prosodic accommodation refers to the tendency of conversational partners to align their speech patterns—pitch, intensity, rate—over time. VibeChecker quantifies this alignment through multiple strategies and derives interpretable metrics.

---

## Pipeline Overview

VibeChecker orchestrates the following steps to analyze prosodic accommodation between two speakers:

1. **Input**: Two-channel WAV or two single-channel WAVs plus a CSV transcript with time-aligned turns.
2. **Feature Extraction**: Use `AudioFeatures` to extract low-level prosodic features (pitch, intensity, rate) for each speaker over frames or turns.
3. **Accommodation Computation**: Apply selected strategy (turn-level, TAMA, or hybrid) to pair or window feature series and compute synchrony and convergence time-series.
4. **State Classification**: Label each turn or frame as synchronized, asynchronized, converging, or diverging based on thresholds.
5. **High-Level Metrics**: Aggregate time-series into summary metrics—mean/SD of synchrony, convergence slopes, state durations, and concurrent state proportions.
6. **Outputs**:

   * **CSV Report**: Summary of metrics and state proportions.
   * **Pickled Data**: Raw time-series for further analysis.
   * **PDF/PNG Visualizations**: LOESS-smoothed plots of feature trajectories and state timelines.

---

## Accomodation Methods Comparison

| Method                    | How computed                                                             | What it returns                                 | What it captures                          | Pros                                              | Cons                                                        | When to use                                        | Most similar to                    |
| ------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------- | ----------------------------------------- | ------------------------------------------------- | ----------------------------------------------------------- | -------------------------------------------------- | ---------------------------------- |
| **Turn-level “turn”**     | Correlate A’s feature at turn i−1 with B’s at turn i across all turns    | One Pearson r per feature                       | Overall lag-1 turn-taking synchrony       | Very simple; no extra parameters                  | No time course; ignores within-turn variation               | Quick high-level check of turn-by-turn synchrony   | **Combined**                       |
| **Turn-level “dynamic”**  | Slide window of consecutive turns, compute Pearson r each window         | Time-series of r values with timestamps         | Local synchrony fluctuations              | Time-resolved; shows ups and downs                | Sensitive to window/hop choices; can be noisy               | Examining how synchrony evolves during interaction | **TAMA (dynamic)**                 |
| **Turn-level “combined”** | Average of turn-level r and mean dynamic-window r                        | One averaged Pearson r per feature              | Blend of static and dynamic synchrony     | Balances coarse and fine information              | Arbitrary equal weighting; still no time course             | Single metric reflecting both views                | Bridges **turn** & **dynamic**     |
| **TAMA**                  | Extract features in overlapping windows; compute Pearson r in each frame | Detailed time-series of r values and timestamps | Continuous synchrony at regular intervals | Robust to segmentation errors; uniform resolution | Ignores utterance boundaries; window choice critical        | Uniform analysis independent of turn boundaries    | **Turn-level dynamic**             |
| **HYBRID**                | Extend windows to utterance boundaries then sliding-window correlation   | Time-series of r values and timestamps          | Dynamics that respect utterance structure | Captures utterance-level prosody; multiscale      | Requires good utterance segmentation; more complex pipeline | When utterance-level prosody matters               | Mix of **turn-dynamic** & **TAMA** |

### Similarities & Differences

All methods use Pearson *r* as their core metric. Dynamic, TAMA, and HYBRID return time-series of *r* values plus timestamps (and optional phase counts); turn and combined return a single scalar per feature.

* **Window Basis**: TAMA uses uniform time frames; dynamic slides over turn indices; HYBRID aligns to utterance boundaries.
* **Granularity**: Turn and combined yield scalars; dynamic, TAMA, HYBRID yield time-series.
* **Segmentation Sensitivity**: TAMA is robust to transcript errors; HYBRID requires accurate utterance segmentation; turn methods ignore within-turn variations.

### When to Pick Which

* **Turn-level “turn”**: Quick, interpretable overall check of turn-by-turn synchrony.
* **Turn-level “dynamic”**: Examine local fluctuations of synchrony across the conversation.
* **Turn-level “combined”**: One summary metric blending static and dynamic views.
* **TAMA**: Uniform, transcript-independent analysis; ideal when turn boundaries are unreliable.
* **HYBRID**: Utterance-grounded dynamic analysis; use when prosody at the utterance scale matters.

Configure specifics—window size, hop length, thresholds—via the `AccomConfig` class.

---

## State Classification

VibeChecker labels each frame or turn based on synchrony and convergence metrics relative to configurable thresholds:

* **Synchronized**: synchrony score ≥ threshold and |A−B| distance decreasing or stable.
* **Asynchronized**: synchrony score < threshold and |A−B| distance stable.
* **Converging**: slope of |A−B| over time < −threshold (features moving closer).
* **Diverging**: slope of |A−B| over time > threshold (features moving apart).

Thresholds (e.g., synchrony and convergence) are set in `AccomConfig` (`state_thresh`). You can also adjust LOESS smoothing for robust state detection.

---

### Feature Strategy Classes

| Class Name            | Purpose                                                                                 | Calculation                                                                                           |             |                         |
| --------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ----------- | ----------------------- |
| `TurnSynchrony`       | Turn-level synchrony: correlation of successive turns.                                  | PearsonCorr(Aᵢ₋₁, Bᵢ) over all valid i ≥ 1.                                                           |             |                         |
| `DynamicSynchrony`    | Dynamic synchrony: sliding-window correlation over feature trajectories.                | Sliding-window PearsonCorr over paired feature series within each window defined by `window` & `hop`. |             |                         |
| `CombinedSynchrony`   | Unified synchrony metric combining turn-level and dynamic.                              | Aggregate (e.g., mean) of turn-level and dynamic synchrony scores for each segment or session.        |             |                         |
| `ConvergenceStrategy` | Convergence: directional alignment of feature values over time.                         | Slope of linear regression of                                                                         | A\_t − B\_t | against time indices t. |
| `StateStrategy`       | Frame/turn-level state classification into synchrony/asynchrony/convergence/divergence. | Compare synchrony and convergence values to threshold: assign state based on >/- threshold criteria.  |             |                         |
| `ConcurrentStrategy`  | Concurrent states: joint consistency across all features.                               | Fraction of windows where every feature’s state label matches across speakers within that window.     |             |                         |

Configure specifics—window size, hop length, thresholds—via the `AccomConfig` class.—window size, hop length, thresholds—via the `AccomConfig` class.

---

## Generated Metrics & States

VibeChecker computes:

* **Synchrony**

  * *Turn-level*: Pearson correlation between adjacent turns.
  * *Dynamic (TAMA)*: Sliding-window correlation over feature time-series.
  * *Combined*: Aggregate of both.
* **Convergence/Divergence**

  * Slope of inter-speaker distance changes (|A−B| over time).
* **State Classification**

  * Labels each frame/turn as synchronized, asynchronized, converging, or diverging.
* **State Durations**

  * Total and mean durations for each state.
* **Concurrent States**

  * Fraction of windows where *all* features share the same state.

**Outputs**

* **CSV Reports**: Mean, SD, and proportions of metrics and states.
* **Pickle Files**: Raw time-series data under `results/.../pickles/`.
* **Visualizations**: LOESS-smoothed plots of feature trajectories, synchrony phases, and state timelines.

---

## Audio Features

Extracted by `AudioFeatures` in `audio_features/audio_features.py`:

* **Pitch (F0)**: min, max, mean, median, SD, range, 80th-percentile.
* **Intensity**: mean & SD of amplitude contour.
* **Articulation Rate**: syllable nuclei per second (De Jong & Wempe, 2009).
* **csi**: 

Use `extract_all()` to retrieve all features or `extract(keys=[...])` for a subset.

---

## Pipeline & CLI Options

The `prosodic_accomodation_pipeline.py` script provides a Click-based CLI.

### `run` command

```bash
Usage: cli run [OPTIONS] AUDIO_PATH TRANSCRIPT_PATH
```

Options:

* `-t, --accommodation-type [turn_level|tama|hybrid]` (default: turn\_level)
* `-f, --features TEXT` (comma-separated; default: all)
* `-r, --results-path PATH` (default: `results/`)
* `--no-viz/--viz` (toggle visualizations; default: viz enabled)
* `-v, --verbose` (verbose logs)
* `--synchrony-mode [turn|dynamic|combined]` (default: turn)
* `--win-frames INTEGER` (window length; default: 10 frames)
* `--hop-frames INTEGER` (hop length; default: 5 frames)
* `--state-thresh FLOAT` (classification threshold; default: 0.5)
* `--loess-frac FLOAT` (LOESS smoothing fraction; default: 0.3)

Workflow:

1. Build `AccomConfig` (calculates frame\_duration).
2. Extract feature time-series via `.get_accommodation()`.
3. Compute synchrony, convergence, state metrics.
4. Save CSV summary, pickles, and PNG plots.

### `list-features` command

```bash
$ cli list-features
```

Prints all available feature keys:

````
mean_f0
sd_f0
mean_intensity
syllables_per_second
...
``` 

---
## Installation
```bash
git clone https://github.com/your-username/vibechecker.git
cd vibechecker
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
````

---

## File Structure

```
vibechecker/
├── accomodation_types/       # Turn-level, TAMA, Hybrid
├── audio_features/           # Feature extraction
├── accom_features/           # Config & FeatureStrategy
├── data/                     # Example audio & transcripts
├── data_types/               # Audio & transcript wrappers
├── prosodic_accomodation_pipeline.py  # CLI entrypoint
└── tests/                    # Unit tests
```

---

## Usage

### Data Preparation

1. Place speaker audio (`.wav`) under `data/audio/`.
2. Add transcript CSV under `data/transcripts/` with columns: `start,end,speaker,text`.

### Running the Pipeline

```bash
python prosodic_accomodation_pipeline.py \
  data/audio/s1_s2.wav data/transcripts/s1_s2.csv \
  --accommodation-type hybrid \
  --features mean_f0,syllables_per_second \
  --results-path results/ \
  --synchrony-mode combined \
  --win-frames 20 --hop-frames 10 \
  --state-thresh 0.6 --loess-frac 0.2 \
  --no-viz
```

### Listing Features

```bash
python prosodic_accomodation_pipeline.py list-features
```

---

## Programmatic API

```python
from accom_features.accom_config import AccomConfig
from accomodation_types.hybrid_prosodic_acomodation import HybridProsodicAccomodation
from accom_features.feature_strategy import ConvergenceStrategy, DynamicSynchrony

# 1. Extract time-series
ac = HybridProsodicAccomodation(
    audio_path="data/audio/s1.wav",
    transcript_csv="data/transcripts/s1.csv",
    requested_features=["mean_f0","sd_f0"],
    verbose=False
)
time_series = ac.get_accommodation()

# 2. Configure metrics
cfg = AccomConfig(
    frame_duration=ac.frame_duration,
    window=15,
    hop=5,
    thresh=0.5,
    synchrony_mode="dynamic"
)

# 3. Compute metrics
conv = ConvergenceStrategy(
    time_series['mean_f0'][:,0],
    time_series['mean_f0'][:,1],
    cfg
).compute()
sync = DynamicSynchrony(
    time_series['mean_f0'][:,0],
    time_series['mean_f0'][:,1],
    cfg
).compute()
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
