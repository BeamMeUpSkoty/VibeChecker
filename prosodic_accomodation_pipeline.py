#!/usr/bin/env python3

import os
import pickle
import click
import csv as _csv
import numpy as np
from audio_features.audio_features import AudioFeaturesOptimized

from accomodation_types.turn_level_prosodic_acommodation import TurnLevelProsodicAccomodation
from accomodation_types.hybrid_prosodic_acomodation import HYBRIDProsodicAcommodation
from accomodation_types.tama_prosodic_accomodation import TAMAProsodicAcommodation

@click.group()
@click.version_option(version='1.0', prog_name='prosodic-accommodation')
def cli():
    """
    Prosodic Accommodation CLI

    Use 'run' to execute analyses or 'list-features' to see available prosodic features.
    """
    pass

@cli.command()
@click.argument('audio_path', type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.argument('transcript_path', type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.option('--accommodation-type', '-t',
              type=click.Choice(['turn_level', 'hybrid', 'tama']),
              default='turn_level', show_default=True,
              help='Type of accommodation analysis to perform')
@click.option('--features', '-f', default='',
              help='Comma-separated list of prosodic features (default: all)')
@click.option('--results-path', '-r', default='results/', show_default=True,
              type=click.Path(), help='Base directory to save results and plots')
@click.option('--no-viz', 'visualize', flag_value=False, default=True,
              help='Skip generating visualizations')
@click.option('--verbose', '-v', is_flag=False,
              help='Enable verbose logging')
@click.option('--synchrony-mode',
              type=click.Choice(['turn', 'dynamic', 'combined']),
              default='turn', show_default=True,
              help='(Turn-level only) synchrony mode')
@click.option('--win-frames', type=int, default=10, show_default=True,
              help='Sliding-window length in frames')
@click.option('--hop-frames', type=int, default=5, show_default=True,
              help='Sliding-window hop in frames')
@click.option('--state-thresh', type=float, default=0.5, show_default=True,
              help='Threshold for synchrony/asynchrony and convergence/divergence')
@click.option('--loess-frac', type=float, default=0.3, show_default=True,
              help='LOESS smoothing fraction for plots')
def run(
    audio_path,
    transcript_path,
    accommodation_type,
    features,
    results_path,
    visualize,
    verbose,
    synchrony_mode,
    win_frames,
    hop_frames,
    state_thresh,
    loess_frac
):
    """
    Execute prosodic accommodation analysis on a single file or batch in directories.

    AUDIO_PATH: path to WAV file or directory of WAVs
    TRANSCRIPT_PATH: path to CSV file or directory of CSVs
    """
    # Parse features list
    feats = None
    if features:
        feats = [f.strip() for f in features.split(',') if f.strip()]

    # Determine file pairs
    if os.path.isdir(audio_path) and os.path.isdir(transcript_path):
        audio_files = [f for f in os.listdir(audio_path)
                       if f.lower().endswith('.wav')]
        file_pairs = []
        for wav in audio_files:
            base = os.path.splitext(wav)[0]
            wav_file = os.path.join(audio_path, wav)
            csv_file = os.path.join(transcript_path, base + '.csv')
            if os.path.exists(csv_file):
                file_pairs.append((wav_file, csv_file))
        if not file_pairs:
            raise click.ClickException(
                'No matching WAV/CSV pairs found in directories')
    else:
        file_pairs = [(audio_path, transcript_path)]

    # Prepare summary CSV path
    common_suffix = f"{accommodation_type}"
    if accommodation_type == 'turn_level':
        common_suffix += f"_{synchrony_mode}"
    common_suffix += f"_w{win_frames}_h{hop_frames}_th{state_thresh}_loess{loess_frac}"
    summary_csv = os.path.join(results_path, f"{common_suffix}_summary.csv")
    summary_rows = []

    # Process each file pair
    for wav_path, csv_path in file_pairs:
        audio_name = os.path.splitext(os.path.basename(wav_path))[0]
        # Create output directories
        call_dir = os.path.join(results_path, common_suffix)
        img_dir = os.path.join(call_dir, 'images')
        pkl_dir = os.path.join(call_dir, 'pickles')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(pkl_dir, exist_ok=True)

        # Instantiate analyzer
        if accommodation_type == 'turn_level':
            ac = TurnLevelProsodicAccomodation(
                audio_path=wav_path,
                transcript_csv=csv_path,
                requested_features=feats,
                verbose=verbose
            )
            ac.synchrony_mode = synchrony_mode
            conv = ac.get_convergence()
            sync = ac.get_synchrony()
            if visualize:
                plot_path = os.path.join(img_dir, f"{audio_name}_turn.png")
                ac.get_visualization(
                    output_path=plot_path,
                    loess_frac=loess_frac,
                    window=win_frames,
                    hop=hop_frames,
                    thresh=state_thresh
                )
        else:
            PipelineClass = (
                HYBRIDProsodicAcommodation if accommodation_type == 'hybrid'
                else TAMAProsodicAcommodation
            )
            ac = PipelineClass(
                audio_path=wav_path,
                transcript_csv=csv_path,
                requested_features=feats,
                verbose=verbose
            )
            conv = ac.get_convergence()
            sync = ac.get_synchrony_features()
            if visualize:
                plot_path = os.path.join(img_dir, f"{audio_name}_{accommodation_type}.png")
                ac.get_visualization(
                    output_path=plot_path,
                    loess_frac=loess_frac,
                    window=win_frames,
                    hop=hop_frames,
                    thresh=state_thresh
                )

        # Save metrics pickle
        metrics_path = os.path.join(pkl_dir, f"{audio_name}_{accommodation_type}_metrics.pkl")
        with open(metrics_path, 'wb') as f:
            pickle.dump({'convergence': conv, 'synchrony': sync}, f)

        # Build summary row
        row = {
            'audio_path': wav_path,
            'transcript_path': csv_path,
            'duration': getattr(ac, 'duration', None),
            'features': '|'.join(feats) if feats else ''
        }
        # Add convergence
        for feat, val in conv.items():
            row[f'conv_{feat}'] = val
        # Add synchrony metrics
        for feat, arr in sync.items():
            if isinstance(arr, dict) and 'r_values' in arr:
                series = arr['r_values']
                row[f'sync_{feat}_mean'] = np.mean(series)
                row[f'sync_{feat}_n'] = len(series)
            else:
                row[f'sync_{feat}'] = arr
        summary_rows.append(row)

        click.echo(f"Processed {audio_name}, results in {call_dir}")

    # Write summary CSV
    if summary_rows:
        with open(summary_csv, 'w', newline='') as f:
            writer = _csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        click.echo(f"Summary CSV written to {summary_csv}\n")

@cli.command('list-features')
def list_features():
    """
    List available prosodic features from the audio_features module.
    """
    dummy = AudioFeaturesOptimized(array=np.zeros(100), sr=16000)
    try:
        feats = dummy.get_feature_list()
    except AttributeError:
        feats = list(dummy.extract([]).keys())
    click.echo("Available features:")
    for f in feats:
        click.echo(f" - {f}")

if __name__ == '__main__':
    cli()
