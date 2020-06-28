import os
import argparse

parser = argparse.ArgumentParser(description="run Classification.py")
parser.add_argument("--test_image_path", type=str, help="Path to test image")
args = parser.parse_args()

os.system(
    f"python src\\Classification.py --binary_model random_forest --multiclass_model svc --mask_predict_model "
    f"random_forest --sampler smotetomek --test_image_path {args.test_image_path} "
)
