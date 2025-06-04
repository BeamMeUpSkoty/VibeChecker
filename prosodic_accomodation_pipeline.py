'''
created on April 3, 2023

@author: hali02
'''
import os
import fire

from accomodation_types.turn_level_prosodic_acommodation import TurnLevelProsodicAccomodation
from accomodation_types.hybrid_prosodic_acomodation import HYBRIDProsodicAcommodation
from accomodation_types.tama_prosodic_accomodation import TAMAProsodicAcommodation

class ProsodicAccommodation(object):
    """
    Wrapper class to run different prosodic accommodation pipelines.

    Usage (CLI):
      python prosodic_accommodation_pipeline.py \
        --audio_path path/to/audio.wav \
        --transcript_csv path/to/transcript.csv \
        --results_path path/to/results_dir \
        --accommodation_type turn_taking \
        --features mean_f0,sd_f0,mean_intensity \
        --visualize True \
        --verbose False
    """

    def __init__(
        self,
        audio_path: str,
        transcript_csv: str,
        results_path: str = "",
        accommodation_type: str = "turn_level",
        features: str = "",
        visualize: bool = True,
        verbose: bool = False,
    ):
        """
        :param audio_path:         Path to the mixed-speaker WAV file.
        :param transcript_csv:     Path to CSV with columns [start,end,text,speaker].
        :param results_path:       Directory where plots/results will be saved (optional).
        :param accommodation_type: One of {'turn_taking','hybrid'}.
        :param features:           Comma-separated list of feature keys from AudioFeatures.extract(),
                                  e.g. "mean_f0,sd_f0,mean_intensity". If empty, defaults are used.
        :param visualize:          If True, call get_visualization(...).
        :param verbose:            If True, pass verbose=True into AudioFeatures.extract().
        """
        self.audio_path = audio_path
        self.transcript_csv = transcript_csv
        self.results_path = results_path.rstrip("/")  # remove trailing slash if present
        self.accommodation_type = accommodation_type.lower()
        self.visualize = visualize
        self.verbose = verbose

        # at the top of prosodic_accomodation_pipeline.py __init__:
        if isinstance(features, (list, tuple)):
            # join a tuple/list into a comma string
            features_str = ",".join(features)
        else:
            features_str = features

        self.requested_features = [f.strip() for f in features_str.split(",") if f.strip()]

    def prosodic_accommodation_pipeline(self) -> None:
        """
        Run the chosen prosodic accommodation pipeline, print convergence/synchrony,
        and optionally visualize+save plots.
        """
        if self.accommodation_type == "turn_level":
            print("=== Running Turn-Level Prosodic Accommodation ===\n")

            # Instantiate with requested_features (or None to use default features)
            TT = TurnLevelProsodicAccomodation(
                audio_path=self.audio_path,
                transcript_csv=self.transcript_csv,
                requested_features=self.requested_features,
                verbose=self.verbose,
            )

            # Compute accommodation details
            #accom_dict = TT.get_accommodation()
            conv = TT.get_convergence()    # dict: { feature: r_convergence }
            sync = TT.get_synchrony()      # dict: { feature: r_synchrony }

            #  Print convergence & synchrony
            print("=== Convergence (per feature) ===")
            for feat, r_val in conv.items():
                print(f"  {feat}:   r_convergence = {r_val:.4f}")
            print("\n=== Synchrony (per feature) ===")
            for feat, r_val in sync.items():
                print(f"  {feat}:   r_synchrony   = {r_val:.4f}")

            # Optionally visualize (and save to results_path)
            if self.visualize:
                save_png = None
                if self.results_path:
                    os.makedirs(self.results_path, exist_ok=True)
                    save_png = os.path.join(self.results_path, "turn_level_accommodation.png")
                TT.get_visualization(output_path=save_png)

        elif self.accommodation_type == "hybrid":
            print("=== Running Hybrid Prosodic Accommodation ===\n")

            HPA = HYBRIDProsodicAcommodation(
                audio_path=self.audio_path,
                transcript_csv=self.transcript_csv,
                requested_features=self.requested_features,
                window_len=20.0,
                hop=10.0,
                verbose=self.verbose,
            )

            #accom_dict = HPA.get_accommodation()
            conv = HPA.get_convergence() # dict: { feature: r_convergence }
            sync_dict = HPA.get_synchrony(sync_window=5)  # dict: { feature: np.ndarray([...]) }

            print("=== Convergence (per feature) ===")
            for feat, r_val in conv.items():
                print(f"  {feat}:   r_convergence = {r_val:.4f}")
            print("\n=== Synchrony (summary per feature) ===")
            for feat, arr in sync_dict.items():
                mean_r = float(arr.mean()) if arr.size > 0 else 0.0
                print(f"  {feat}:   mean_r_synchrony = {mean_r:.4f}   (n={arr.size})")

            if self.visualize:
                save_png = None
                if self.results_path:
                    os.makedirs(self.results_path, exist_ok=True)
                    save_png = os.path.join(self.results_path, "hybrid_accommodation.png")
                HPA.get_visualization(output_path=save_png)

        elif self.accommodation_type == "tama":
            print("=== Running TAMA Prosodic Accommodation ===\n")

            TPA = TAMAProsodicAcommodation(
                audio_path=self.audio_path,
                transcript_csv=self.transcript_csv,
                requested_features=self.requested_features,
                window_len=20.0,
                hop=10.0,
                verbose=self.verbose,
            )

            #accom_dict = TPA.get_accommodation()
            conv = TPA.get_convergence() # dict: { feature: r_convergence }
            sync_dict = TPA.get_synchrony(sync_window=5) # dict: { feature: np.ndarray([...]) }

            print("=== Convergence (per feature) ===")
            for feat, r_val in conv.items():
                print(f"  {feat}:   r_convergence = {r_val:.4f}")
            print("\n=== Synchrony (summary per feature) ===")
            for feat, arr in sync_dict.items():
                mean_r = float(arr.mean()) if arr.size > 0 else 0.0
                print(f"  {feat}:   mean_r_synchrony = {mean_r:.4f}   (n={arr.size})")

            if self.visualize:
                save_png = None
                if self.results_path:
                    os.makedirs(self.results_path, exist_ok=True)
                    save_png = os.path.join(self.results_path, "hybrid_accommodation.png")
                TPA.get_visualization(output_path=save_png)

        else:
            raise ValueError(
                f"Unknown accommodation_type: {self.accommodation_type}. "
                f"Choose from ['turn_level','hybrid']."
            )

        print("\n=== Pipeline complete ===\n")
        return


if __name__ == "__main__":
    fire.Fire(ProsodicAccommodation)
