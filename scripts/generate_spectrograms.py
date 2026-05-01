import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stda_net.config import load_yaml_config
from stda_net.preprocessing import PreprocessConfig
from stda_net.spectrograms import (
    SpectrogramConfig,
    SpectrogramTransformer,
    process_shhs,
    process_sleepedf,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/generate_spectrograms.yaml")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    pre_cfg = PreprocessConfig(**cfg.get("preprocessing", {}))
    spec_cfg = SpectrogramConfig(target_fs=pre_cfg.target_fs, **cfg.get("spectrogram", {}))
    transformer = SpectrogramTransformer(spec_cfg)

    dataset = cfg["dataset"].lower()
    output_root = cfg["output_root"]
    max_subjects = cfg.get("max_subjects", None)

    if dataset in ("sleepedf", "all"):
        process_sleepedf(cfg["sleepedf_dir"], output_root, max_subjects, pre_cfg, transformer)

    if dataset in ("shhs1", "all"):
        process_shhs(cfg["shhs_edf_dir"], cfg["shhs_label_dir"], output_root, "shhs1", max_subjects, pre_cfg, transformer)

    if dataset in ("shhs2", "all"):
        process_shhs(cfg["shhs_edf_dir"], cfg["shhs_label_dir"], output_root, "shhs2", max_subjects, pre_cfg, transformer)


if __name__ == "__main__":
    main()
