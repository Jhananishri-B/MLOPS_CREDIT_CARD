import pandas as pd

def evaluate_results():
    results = pd.read_csv("models/results.csv")
    print("\nModel Evaluation Results:")
    print(results)

if __name__ == "__main__":
    evaluate_results()
