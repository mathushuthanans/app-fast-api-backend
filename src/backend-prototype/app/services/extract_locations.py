import pathlib

def extract():
    """Get the ml_models directory path using proper package imports."""
    try:
        # Get the module's __file__ attribute first
        from ml_models import __file__ as ml_models_file
        MODEL_DIR = pathlib.Path(ml_models_file).parent
    except (ImportError, AttributeError):
        # Fallback for development without installation
        PROJECT_ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
        MODEL_DIR = PROJECT_ROOT_DIR / 'ml_models'
    
    print("MODEL_DIR:", MODEL_DIR)
    return MODEL_DIR