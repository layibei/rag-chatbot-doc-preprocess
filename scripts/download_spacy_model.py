import os
import spacy
import shutil
from pathlib import Path

from config.common_settings import CommonConfig


def download_spacy_model():
    """Download spaCy model and store it locally"""
    try:
        # Download the model
        spacy.cli.download("en_core_web_md")
        
        # Get the model path
        nlp = spacy.load("en_core_web_md")
        model_path = Path(nlp.path)
        
        # Create local models directory
        local_models_dir = Path("../models/spacy")
        local_models_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model to local directory
        local_model_path = local_models_dir / "en_core_web_md"
        if local_model_path.exists():
            shutil.rmtree(local_model_path)
        shutil.copytree(model_path, local_model_path)
        
        print(f"Successfully downloaded and stored spaCy model at {local_model_path}")
        
    except Exception as e:
        print(f"Error downloading spaCy model: {str(e)}")
        raise

if __name__ == "__main__":
    CommonConfig().setup_proxy()
    download_spacy_model() 