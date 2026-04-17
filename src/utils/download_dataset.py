from pathlib import Path

import dotenv
import kagglehub

# for the kaggle api key
dotenv.load_dotenv(dotenv.find_dotenv())

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
print(DATA_DIR)
# kagglehub.competition_download("nml-2026", output_dir=str(DATA_DIR))
