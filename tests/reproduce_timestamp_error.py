import pandas as pd
import pm4py


def reproduce_error():
    print("--- Reproducing Timestamp Error ---")

    # Create a DF where timestamp column is object dtype containing pd.Timestamp objects
    # This simulates what might happen after loading/cleaning
    df = pd.DataFrame(
        {
            "case": [1, 1, 2],
            "act": ["A", "B", "A"],
            "ts": [
                pd.Timestamp("2023-01-01 10:00"),
                pd.Timestamp("2023-01-01 11:00"),
                pd.Timestamp("2023-01-01 12:00"),
            ],
        }
    ).astype(object)

    print(f"Initial dtype: {df['ts'].dtype}")
    print(f"First element type: {type(df['ts'][0])}")

    try:
        # This is what I have in analysis.py
        ts_data = df["ts"]
        df["ts"] = pd.to_datetime(ts_data, errors="coerce")
        print("pd.to_datetime success")

        # This is the next step
        df["ts"] = df["ts"].dt.to_pydatetime()
        print("dt.to_pydatetime success")

        # pm4py format
        formatted_df = pm4py.format_dataframe(
            df, case_id="case", activity_key="act", timestamp_key="ts"
        )
        print("pm4py.format_dataframe success")

    except Exception as e:
        print(f"FAILED: {e}")


if __name__ == "__main__":
    reproduce_error()
