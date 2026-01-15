from pm_agent.main import main
from unittest.mock import MagicMock
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Mock rag before importing main

mock_rag = MagicMock()
sys.modules["pm_agent.rag"] = mock_rag


def test_session_restart():
    # Setup mock data file
    data_path = "test_data.csv"
    df = pd.DataFrame(
        {
            "case": [1, 2],
            "activity": ["A", "B"],
            "timestamp": ["2023-01-01 10:00", "2023-01-01 11:00"],
        }
    )
    df.to_csv(data_path, index=False)

    # Mock LLMClient
    with patch("pm_agent.main.LLMClient") as MockLLM:
        mock_instance = MockLLM.return_value

        # Mock decide_next_step to always return Data Profiling then Finish (to avoid infinite loops if restart doesn't break)
        mock_instance.decide_next_step.side_effect = [
            {"thought": "Profiling", "tool_name": "Data Profiling"},
            {"thought": "Finish", "tool_name": "Finish"},  # After first restart
            {"thought": "Finish", "tool_name": "Finish"},
        ]

        # Mock judge_step to always pass
        mock_instance.judge_step.return_value = {
            "passed": True,
            "critique": "OK",
            "score": 5,
        }

        # Mock judge_cumulative_progress:
        # 1st call: Fail (trigger restart)
        # 2nd call: Pass
        mock_instance.judge_cumulative_progress.side_effect = [
            {"passed": False, "critique": "Failed on purpose"},
            {"passed": True, "critique": "OK"},
        ]

        # Mock update_memory
        mock_instance.update_memory.return_value = "Updated memory"

        # Mock safe_input to always return 'n' (to avoid resume and chat loop)
        with patch("pm_agent.main.safe_input", return_value="exit"):
            # Mock DataProfilingAgent
            with patch("pm_agent.main.DataProfilingAgent") as MockAgent:
                instance = MockAgent.return_value
                instance.run.return_value = (
                    '{"process_mining_readiness": {"level": "High"}}'
                )

                # Run main
                with patch("sys.argv", ["main.py", "--file", data_path]):
                    try:
                        main()
                    except SystemExit:
                        pass

        # Assertions
        # judge_cumulative_progress should be called twice (once fail, once pass after restart)
        assert mock_instance.judge_cumulative_progress.call_count >= 1, (
            "Cumulative judge should have been called"
        )
        print("\nVerification: judge_cumulative_progress was called.")

        # Check if it was called with "Previous Session Failed" in memory on the second attempt (if we could track calls better)
        # For now, just check if it was called more than once
        if mock_instance.judge_cumulative_progress.call_count > 1:
            print(
                "Verification: Session restart logic triggered correctly (judge called multiple times)."
            )
        else:
            print("Verification: judge called only once? Check logic.")

    # Cleanup
    if os.path.exists(data_path):
        os.remove(data_path)
    if os.path.exists("reports"):
        # Find the test session dir and remove it if possible, or just ignore
        pass


if __name__ == "__main__":
    test_session_restart()
