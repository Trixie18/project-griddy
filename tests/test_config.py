def test_paths_exist():
    from src.utils.config import DATA_RAW, DATA_PROCESSED, OUTPUTS_FORECASTS
    assert DATA_RAW.exists()
    assert DATA_PROCESSED.exists()
    assert OUTPUTS_FORECASTS.exists()

def test_device_detected():
    from src.utils.config import DEVICE
    assert DEVICE in ("mps", "cuda", "cpu")
    print(f"Running on: {DEVICE}")