import argparse
import sys
import os

# Ensure src is importable
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def main():
    parser = argparse.ArgumentParser(description="Evaluate the function inverse model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    print(f"Evaluating model from {args.checkpoint}...")
    # TODO: Load model and run evaluation

if __name__ == "__main__":
    main()
