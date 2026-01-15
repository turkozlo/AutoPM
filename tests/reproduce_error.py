
import numpy as np
import pandas as pd


def test_ambiguous_truth():
    print("Testing potential ambiguous truth value locations...")

    # Case 1: if series:
    try:
        s = pd.Series([1, 2, 3])
        if s:
            print("if s passed")
    except Exception as e:
        print(f"if s failed: {e}")

    # Case 2: if list:
    try:
        l = [1, 2, 3]
        if l:
            print("if l passed")
    except Exception as e:
        print(f"if l failed: {e}")

    # Case 3: if numpy array:
    try:
        a = np.array([1, 2, 3])
        if a:
            print("if a passed")
    except Exception as e:
        print(f"if a failed: {e}")

    # Case 4: if df.empty:
    try:
        df = pd.DataFrame({'a': [1]})
        if df.empty:
            print("if df.empty passed")
        else:
            print("if df.empty passed (False)")
    except Exception as e:
        print(f"if df.empty failed: {e}")


if __name__ == "__main__":
    test_ambiguous_truth()
