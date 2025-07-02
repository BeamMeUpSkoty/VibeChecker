from dataclasses import dataclass

@dataclass(frozen=True)
class AccomConfig:
    frame_duration: float
    window: int
    hop: int
    thresh: float
    synchrony_mode: str  # "turn", "dynamic", or "combined"
