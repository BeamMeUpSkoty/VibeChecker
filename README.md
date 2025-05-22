# VibeChecker

This Python package calculates prosodic accommodation between two speakers in a conversation. It extracts and compares speech features turn-by-turn or via summary measures.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/prosodic-accommodation.git
   cd prosodic-accommodation
   ```
2. Set up the virtual environment:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # macOS/Linux
    .venv\Scripts\activate     # Windows
    ```
3. Install the Dependencies
    ```commandline
    pip install -r requirements.txt
    ```
   
## File Structure
```
prosodic-accommodation/
    ├── accomodation_types/
        ├── hybrid_prosodic_acomodation.py
        ├── tama_prosodic_accomodation.py
        ├── turn_level_prosodic_acommodation.py
    ├── audio_features/
        ├── audio_features.py
        ├── my_voice_analysis/
        ├── my_voice_analysis_features.py
    ├── data/
    ├── data_types/
        ├── audio_file.py
        ├── transcript_file.py
    ├── prosodic_accomodation_pipeline.py
```

## Usage

### Add the following to data/:

- Audio files (e.g., .wav) for each speaker.
- Transcripts with time-aligned speech segments or turns.

### Run the pipeline
To run the full analysis, use:

   ```commandline
   python prosodic_accomodation_pipeline.py --audio_dir path/to/audio --transcripts path/to/transcripts --output results.csv
   ```
You can also explore other scripts for more specific analyses:

- turn_level_prosodic_acommodation.py — turn-by-turn prosodic alignment
- tama_prosodic_accomodation.py — summary-based matching
hybrid_prosodic_acomodation.py — combines both
