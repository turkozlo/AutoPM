import json

import pandas as pd

from pm_agent.agents.analysis import ProcessAnalysisAgent
from pm_agent.agents.discovery import ProcessDiscoveryAgent


class MockLLM:
    def generate_response(self, prompt, system_prompt):
        return '{"case_id": "case", "activity": "act", "timestamp": "ts"}'


def test_timestamp_robustness():
    print("--- Testing Timestamp Robustness and Column Conflict ---")

    # Create a DF where timestamp is already pd.Timestamp objects (common in pandas)
    # AND it has conflicting pm4py columns
    df = pd.DataFrame({
        'case': [1, 1, 2],
        'act': ['A', 'B', 'A'],
        'ts': pd.to_datetime(['2023-01-01 10:00', '2023-01-01 11:00', '2023-01-01 12:00']),
        'concept:name': ['Old Activity', 'Old Activity', 'Old Activity']  # Conflict
    })

    llm = MockLLM()

    # Test Discovery
    print("Testing Discovery with pd.Timestamp and conflicting columns...")
    agent_disc = ProcessDiscoveryAgent(df, llm)
    res_disc_json = agent_disc.run()
    res_disc = json.loads(res_disc_json)

    if "error" in res_disc:
        print(f"FAILED Discovery: {res_disc['error']}")
    else:
        print("SUCCESS Discovery: Completed without error.")
        # Check if it used 'act' and not 'concept:name'
        if res_disc['pm_columns']['activity'] == 'act':
            print("SUCCESS: Correct activity column identified.")
        else:
            print(f"FAILED: Wrong activity column: {res_disc['pm_columns']['activity']}")

    # Test Analysis
    print("\nTesting Analysis with pd.Timestamp and conflicting columns...")
    agent_anal = ProcessAnalysisAgent(df, llm)
    res_anal_json = agent_anal.run(pm_columns={'case_id': 'case', 'activity': 'act', 'timestamp': 'ts'})
    res_anal = json.loads(res_anal_json)

    if "error" in res_anal:
        print(f"FAILED Analysis: {res_anal['error']}")
    else:
        print("SUCCESS Analysis: Completed without error.")
        print(f"Mean duration: {res_anal['case_duration']['mean']['value']} {res_anal['case_duration']['mean']['unit']}")


if __name__ == "__main__":
    test_timestamp_robustness()
