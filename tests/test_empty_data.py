import pandas as pd
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from pm_agent.agents.profiling import DataProfilingAgent
from pm_agent.agents.discovery import ProcessDiscoveryAgent
from pm_agent.agents.analysis import ProcessAnalysisAgent
from pm_agent.agents.visualization import VisualizationAgent

class MockLLM:
    def generate_response(self, prompt, system_prompt, **kwargs):
        if "candidates" in prompt.lower():
            return '{"process_mining_readiness": {"score": 0, "case_id_candidates": ["A"], "activity_candidates": ["B"], "timestamp_candidates": ["C"]}, "thoughts": "Mock"}'
        if "plan" in prompt.lower() or "candidates" in system_prompt.lower():
             return '[]'
        return "Mock Interpretation"

def test_empty_data():
    df = pd.DataFrame(columns=['A', 'B', 'C'])
    llm = MockLLM()
    output_dir = "test_empty_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("--- Testing Profiling ---")
    agent = DataProfilingAgent(df, llm)
    res = agent.run()
    print(res)

    print("\n--- Testing Discovery ---")
    agent = ProcessDiscoveryAgent(df, llm)
    res = agent.run(pm_columns={'case_id': 'A', 'activity': 'B', 'timestamp': 'C'}, output_dir=output_dir)
    print(res)

    print("\n--- Testing Visualization ---")
    agent = VisualizationAgent(df, llm)
    # VisualizationAgent.run needs a profiling report
    report = {
        "process_mining_readiness": {"case_id_candidates": ["A"], "activity_candidates": ["B"], "timestamp_candidates": ["C"]},
        "columns": {"A": {"dtype": "int", "nan": 0, "unique": 0}, "B": {"dtype": "object", "nan": 0, "unique": 0}, "C": {"dtype": "datetime64[ns]", "nan": 0, "unique": 0}}
    }
    res = agent.run(report, output_dir=output_dir)
    print(res)

    print("\n--- Testing Analysis ---")
    agent = ProcessAnalysisAgent(df, llm)
    res = agent.run(pm_columns={'case_id': 'A', 'activity': 'B', 'timestamp': 'C'}, output_dir=output_dir)
    print(res)

    # Check for png files
    pngs = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    print(f"\nCreated images: {pngs}")
    if pngs:
        print("FAILED: Images were created on empty data!")
        sys.exit(1)
    else:
        print("SUCCESS: No images created on empty data.")

if __name__ == "__main__":
    test_empty_data()
