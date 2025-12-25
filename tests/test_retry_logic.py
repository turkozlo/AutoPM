import pandas as pd
import json
from pm_agent.agents.cleaning import DataCleaningAgent
from pm_agent.agents.discovery import ProcessDiscoveryAgent
from pm_agent.agents.analysis import ProcessAnalysisAgent

class MockLLM:
    def generate_response(self, prompt, system_prompt):
        if "Data Cleaning Agent" in system_prompt:
            return '[{"column": "activity", "action": "drop_row", "reason": "test"}]'
        return '{"case_id": "case", "activity": "activity", "timestamp": "timestamp"}'

def test_retry_logic():
    df = pd.DataFrame({
        'case': [1, 1, 2],
        'activity': ['A', 'B', None],
        'timestamp': ['2023-01-01 10:00', '2023-01-01 11:00', '2023-01-01 12:00']
    })
    
    llm = MockLLM()
    
    # Test Cleaning Agent
    agent = DataCleaningAgent(df, llm)
    profiling_report = {
        'row_count': 3,
        'columns': {'activity': {'nan': 1}},
        'duplicates': 0
    }
    
    print("--- Testing Cleaning Agent Retry ---")
    res1, df1 = agent.run(profiling_report)
    print(f"Run 1 rows: {len(df1)}")
    
    res2, df2 = agent.run(profiling_report)
    print(f"Run 2 rows: {len(df2)}")
    
    assert len(df1) == 2, "Run 1 should have 2 rows"
    assert len(df2) == 2, "Run 2 should have 2 rows (retry worked on original df)"
    assert len(df) == 3, "Original df should remain unchanged"
    print("Cleaning Agent Retry: SUCCESS")

    # Test Discovery Agent
    print("\n--- Testing Discovery Agent Retry ---")
    df_disc = pd.DataFrame({
        'case': [1, 1, 2],
        'activity': ['A', 'B', 'C'],
        'timestamp': ['invalid', '2023-01-01 11:00', '2023-01-01 12:00']
    })
    agent_disc = ProcessDiscoveryAgent(df_disc, llm)
    
    res1 = json.loads(agent_disc.run())
    print(f"Run 1 activities: {res1.get('activities')}")
    
    res2 = json.loads(agent_disc.run())
    print(f"Run 2 activities: {res2.get('activities')}")
    
    assert res1.get('activities') == 2, "Run 1 should have 2 activities (one row dropped due to invalid timestamp)"
    assert res2.get('activities') == 2, "Run 2 should have 2 activities (retry worked on original df)"
    assert len(df_disc) == 3, "Original df should remain unchanged"
    print("Discovery Agent Retry: SUCCESS")

if __name__ == "__main__":
    test_retry_logic()
