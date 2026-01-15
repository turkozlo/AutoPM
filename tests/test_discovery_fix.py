from pm_agent.agents.visualization import VisualizationAgent
from pm_agent.agents.discovery import ProcessDiscoveryAgent
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))


class MockLLM:
    def generate_response(self, prompt, system_prompt):
        if "Columns" in prompt:
            return '{"case_id": "case:concept:name", "activity": "concept:name", "timestamp": "time:timestamp"}'
        return "Mock interpretation"


def test_discovery_fix():
    print("Running verification test for discovery crash fix...")

    # 1. Create a mock dataframe that mimics pm4py-formatted data (which might have caused issues)
    data = {
        "case:concept:name": ["C1", "C1", "C2"],
        "concept:name": ["A", "B", "A"],
        "time:timestamp": ["2023-01-01 10:00", "2023-01-01 11:00", "2023-01-01 12:00"],
    }
    df = pd.DataFrame(data)

    llm = MockLLM()

    # 2. Run VisualizationAgent (should not modify original df)
    print("Testing VisualizationAgent isolation...")
    profiling_report = {
        "process_mining_readiness": {
            "activity_candidates": ["concept:name"],
            "timestamp_candidates": ["time:timestamp"],
            "case_id_candidates": ["case:concept:name"],
        },
        "columns": {
            "concept:name": {"dtype": "object"},
            "time:timestamp": {"dtype": "object"},
            "case:concept:name": {"dtype": "object"},
        },
    }

    vis_agent = VisualizationAgent(df, llm)
    vis_agent.run(profiling_report, output_dir="reports")

    # Check if df was modified (it shouldn't be, but visualization might have converted types in its copy)
    # The original df['time:timestamp'] was object type.
    print(f"Original DF timestamp dtype: {df['time:timestamp'].dtype}")

    # 3. Run ProcessDiscoveryAgent
    print("Testing ProcessDiscoveryAgent with pre-formatted columns...")
    discovery_agent = ProcessDiscoveryAgent(df, llm)
    result_json = discovery_agent.run(output_dir="reports")

    result = json.loads(result_json)
    if "error" in result:
        print(f"FAILED: Discovery returned error: {result['error']}")
        sys.exit(1)

    print("SUCCESS: Discovery completed without errors.")
    print(f"Identified columns: {result['pm_columns']}")

    # 4. Test with string dates that need conversion
    print("Testing ProcessDiscoveryAgent with string dates...")
    df_str = pd.DataFrame(
        {
            "case": ["1", "1", "2"],
            "act": ["Start", "End", "Start"],
            "dt": ["2023-01-01", "2023-01-02", "2023-01-01"],
        }
    )
    discovery_agent_str = ProcessDiscoveryAgent(df_str, llm)
    # Mock LLM will return standard names, but fallback should handle it or we pass them
    pm_cols = {"case_id": "case", "activity": "act", "timestamp": "dt"}
    result_json_str = discovery_agent_str.run(pm_columns=pm_cols, output_dir="reports")

    result_str = json.loads(result_json_str)
    if "error" in result_str:
        print(
            f"FAILED: Discovery with string dates returned error: {result_str['error']}"
        )
        sys.exit(1)

    print("SUCCESS: Discovery with string dates completed.")

    # 5. Test ProcessAnalysisAgent
    print("Testing ProcessAnalysisAgent...")
    from pm_agent.agents.analysis import ProcessAnalysisAgent

    analysis_agent = ProcessAnalysisAgent(df, llm)
    # df is already formatted by DiscoveryAgent in the actual pipeline,
    # but here we test if it handles the same columns.
    analysis_json = analysis_agent.run(
        pm_columns={
            "case_id": "case:concept:name",
            "activity": "concept:name",
            "timestamp": "time:timestamp",
        },
        output_dir="reports",
    )

    analysis_res = json.loads(analysis_json)
    if "error" in analysis_res:
        print(f"FAILED: Analysis returned error: {analysis_res['error']}")
        sys.exit(1)

    print("SUCCESS: Analysis completed without errors.")

    # 6. Test with Duplicate Column Names
    print("Testing with duplicate column names...")
    df_dup = pd.DataFrame(
        [
            ["C1", "A", "2023-01-01", "Extra"],
            ["C1", "B", "2023-01-02", "Extra"],
            ["C2", "A", "2023-01-01", "Extra"],
        ],
        columns=["case", "act", "dt", "dt"],
    )  # Two 'dt' columns

    discovery_agent_dup = ProcessDiscoveryAgent(df_dup, llm)
    pm_cols_dup = {"case_id": "case", "activity": "act", "timestamp": "dt"}
    result_json_dup = discovery_agent_dup.run(
        pm_columns=pm_cols_dup, output_dir="reports"
    )

    result_dup = json.loads(result_json_dup)
    if "error" in result_dup:
        print(
            f"FAILED: Discovery with duplicate columns returned error: {result_dup['error']}"
        )
        sys.exit(1)

    print("SUCCESS: Discovery with duplicate columns completed.")

    print("\nAll verification tests passed!")


if __name__ == "__main__":
    if not os.path.exists("reports"):
        os.makedirs("reports")
    test_discovery_fix()
