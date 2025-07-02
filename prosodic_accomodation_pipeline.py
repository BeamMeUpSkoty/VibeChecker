#!/usr/bin/env python3

import os
import pickle
import click
import csv as _csv
import numpy as np
import matplotlib.pyplot as plt
from audio_features.audio_features import AudioFeatures

from accomodation_types.turn_level_prosodic_acommodation import TurnLevelProsodicAccomodation
from accomodation_types.hybrid_prosodic_acomodation import HYBRIDProsodicAcommodation
from accomodation_types.tama_prosodic_accomodation import TAMAProsodicAcommodation

from accom_features.accom_config import AccomConfig
from accom_features.feature_strategy import (
    ConvergenceStrategy,
    TurnSynchrony,
    DynamicSynchrony,
    CombinedSynchrony,
    ConcurrentFeatureStrategy
)

@click.group()
@click.version_option(version='1.0', prog_name='prosodic-accommodation')
def cli():
    """Prosodic Accommodation CLI"""
    pass

@cli.command()
@click.argument('audio_path', type=click.Path(exists=True))
@click.argument('transcript_path', type=click.Path(exists=True))
@click.option('-t', '--accommodation-type',
              type=click.Choice(['turn_level','hybrid','tama']),
              default='turn_level', show_default=True)
@click.option('-f', '--features', default='',
              help='Comma-separated list of features (default=all)')
@click.option('-r', '--results-path', default='results/', show_default=True,
              type=click.Path(), help='Where to save outputs')
@click.option('--no-viz/--viz', 'visualize', default=True)
@click.option('-v', '--verbose', is_flag=True)
@click.option('--synchrony-mode',
              type=click.Choice(['turn','dynamic','combined']),
              default='turn', show_default=True)
@click.option('--win-frames', type=int, default=10, show_default=True)
@click.option('--hop-frames', type=int, default=5, show_default=True)
@click.option('--state-thresh', type=float, default=0.5, show_default=True)
@click.option('--loess-frac', type=float, default=0.3, show_default=True)

def run(
    audio_path, transcript_path, accommodation_type, features, results_path,
    visualize, verbose, synchrony_mode, win_frames, hop_frames,
    state_thresh, loess_frac
):
    # parse feature list
    feats = [f.strip() for f in features.split(',') if f.strip()] or None

    # build config (frame_duration will be set per-file)
    base_cfg = AccomConfig(
        frame_duration=0.0,
        window=win_frames,
        hop=hop_frames,
        thresh=state_thresh,
        synchrony_mode=synchrony_mode
    )

    # find file pairs
    if os.path.isdir(audio_path) and os.path.isdir(transcript_path):
        bases = [os.path.splitext(f)[0]
                 for f in os.listdir(audio_path) if f.lower().endswith('.wav')]
        pairs = [(os.path.join(audio_path, b+'.wav'),
                  os.path.join(transcript_path, b+'.csv'))
                 for b in bases if os.path.exists(os.path.join(transcript_path, b+'.csv'))]
        if not pairs:
            raise click.ClickException("No matching WAV/CSV pairs found")
    else:
        pairs = [(audio_path, transcript_path)]

    # prepare summary CSV
    suffix = f"{accommodation_type}_{synchrony_mode}" if accommodation_type=='turn_level' else accommodation_type
    suffix += f"_w{win_frames}_h{hop_frames}_th{state_thresh}_loess{loess_frac}"
    summary_csv = os.path.join(results_path, f"{suffix}_summary.csv")
    summary = []

    for wav_path, csv_path in pairs:
        audio_name = os.path.splitext(os.path.basename(wav_path))[0]
        call_dir, img_dir, pkl_dir = [os.path.join(results_path, suffix, sub)
                                      for sub in ('', 'images', 'pickles')]
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(pkl_dir, exist_ok=True)

        # instantiate the right pipeline
        if accommodation_type == 'turn_level':
            ac = TurnLevelProsodicAccomodation(
                audio_path=wav_path, transcript_csv=csv_path,
                requested_features=feats, verbose=verbose
            )
        elif accommodation_type == 'hybrid':
            ac = HYBRIDProsodicAcommodation(
                audio_path=wav_path, transcript_csv=csv_path,
                requested_features=feats, verbose=verbose
            )
        else:  # 'tama'
            ac = TAMAProsodicAcommodation(
                audio_path=wav_path, transcript_csv=csv_path,
                requested_features=feats, verbose=verbose
            )

        # finalize config with actual frame_duration
        cfg = base_cfg.__class__(
            frame_duration=ac.frame_duration,
            window=base_cfg.window,
            hop=base_cfg.hop,
            thresh=base_cfg.thresh,
            synchrony_mode=base_cfg.synchrony_mode
        )

        # extract A/B time series per feature
        accom = ac.get_accommodation()

        # collect summary row
        row = {
            'audio_path': wav_path,
            'transcript_path': csv_path,
            'duration': getattr(ac, 'duration', None),
            'features': '|'.join(ac.requested_features) if ac.requested_features else ''
        }

        for feat in ac.requested_features:
            A = accom[feat][:,0]
            B = accom[feat][:,1]

            # convergence
            conv_r = ConvergenceStrategy(A, B, cfg).compute()
            row[f'conv_{feat}'] = conv_r

            # synchrony
            Strat = {
                'turn':   TurnSynchrony,
                'dynamic': DynamicSynchrony,
                'combined': CombinedSynchrony
            }[cfg.synchrony_mode]
            sync_out = Strat(A, B, cfg).compute()
            if isinstance(sync_out, dict):
                vals = sync_out['r_values']
                row[f'sync_{feat}_mean'] = float(np.mean(vals))
                row[f'sync_{feat}_n']    = int(len(vals))
            else:
                row[f'sync_{feat}']      = float(sync_out)

            # state durations
            states = Strat(A, B, cfg).states()
            durs   = Strat(A, B, cfg).durations()
            for name, val in durs.items():
                row[f'{name}_{feat}'] = val

        # concurrent-state summarization
        masks_per_feat = []
        for feat in ac.requested_features:
            strat = CombinedSynchrony(accom[feat][:,0],
                                     accom[feat][:,1],
                                     cfg)
            states = strat.states()
            masks_per_feat.append({
                'synchrony': np.isin(states, [2, 4]),
                'asynchrony': np.isin(states, [5, 7]),
                'convergence': np.isin(states, [3, 4]),
                'divergence': np.isin(states, [6, 7]),
            })

        hop_seconds = cfg.hop * ac.frame_duration
        concurrent = ConcurrentFeatureStrategy(masks_per_feat,
                                               hop_seconds)

        # --- visualize when all features moved in lockstep ---
        # first build the four “all agree” masks:
        all_sync  = np.logical_and.reduce([m['synchrony']   for m in masks_per_feat])
        all_async = np.logical_and.reduce([m['asynchrony'] for m in masks_per_feat])
        all_conv  = np.logical_and.reduce([m['convergence'] for m in masks_per_feat])
        all_div   = np.logical_and.reduce([m['divergence']  for m in masks_per_feat])

        # time‐axis in seconds
        times = np.arange(len(all_sync)) * hop_seconds

        plt.figure()
        plt.step(times, all_sync,  where='post')
        plt.step(times, all_async, where='post')
        plt.step(times, all_conv,  where='post')
        plt.step(times, all_div,   where='post')
        plt.xlabel('Time (s)')
        plt.ylabel('All‐feature lockstep (0/1)')
        plt.legend(['synchrony','asynchrony','convergence','divergence'])
        plt.title(f'Concurrent prosodic states – {audio_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, f"{audio_name}_concurrent_states.png"))
        plt.close()
        # --- end visualization ---

        conc_durs  = concurrent.durations()
        row.update(conc_durs)

        # pickle raw metrics
        with open(os.path.join(pkl_dir, f"{audio_name}_{accommodation_type}_metrics.pkl"), 'wb') as pf:
            pickle.dump({'convergence': row, 'synchrony': sync_out}, pf)

        # visualize
        if visualize:
            plot_path = os.path.join(img_dir, f"{audio_name}.png")
            ac.get_visualization(
                output_path=plot_path,
                loess_frac=loess_frac,
                window=win_frames,
                hop=hop_frames,
                thresh=state_thresh
            )

        summary.append(row)
        click.echo(f"Processed {audio_name}")

    # write CSV
    if summary:
        with open(summary_csv, 'w', newline='') as f:
            w = _csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        click.echo(f"Summary CSV: {summary_csv}")

@cli.command('list-features')
def list_features():
    AF = AudioFeatures(array=np.zeros(100), sr=16000)
    try:
        feats = AF.get_feature_list()
    except AttributeError:
        feats = list(AF.extract([]).keys())
    click.echo("Available features:")
    for f in feats:
        click.echo(f" - {f}")

if __name__ == '__main__':
    cli()
