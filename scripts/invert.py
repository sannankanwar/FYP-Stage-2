import argparse
import sys
import os

# Ensure src is importable
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def main():
    parser = argparse.ArgumentParser(description="Run inference using the inverted function model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--y-val", type=float, required=True, help="Input y value to find x for")
    args = parser.parse_args()

    print(f"Finding inverse for y={args.y_val}...")
    # TODO: Load model and predict x

if __name__ == "__main__":
    main()
