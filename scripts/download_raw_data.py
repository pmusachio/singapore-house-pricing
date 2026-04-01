from pathlib import Path

DATASET = "darrylljk/singapore-hdb-resale-flat-prices-2017-2024"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    try:
        import kaggle
    except ImportError as exc:
        message = (
            "The Kaggle client is not installed. Run `pip install kaggle` or "
            "`pip install .[download]`, configure your Kaggle credentials, and "
            "then rerun this script."
        )
        raise SystemExit(message) from exc

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    kaggle.api.dataset_download_files(DATASET, path=str(OUTPUT_DIR), unzip=True)
    print(f"Downloaded dataset to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
