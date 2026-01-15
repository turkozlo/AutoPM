import sys
import io
from unittest.mock import MagicMock, patch
from pm_agent.main import print_used_tools


def test_print_used_tools():
    # Test case 1: Empty set
    captured_output = io.StringIO()
    sys.stdout = captured_output
    print_used_tools(set())
    sys.stdout = sys.__stdout__
    assert captured_output.getvalue() == ""

    # Test case 2: Single tool
    used_tools = {"Data Profiling"}
    captured_output = io.StringIO()
    sys.stdout = captured_output
    print_used_tools(used_tools)
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "🛠️ ИСПОЛЬЗОВАННЫЕ ИНСТРУМЕНТЫ" in output
    assert "1. Data Profiling" in output

    # Test case 3: Multiple tools, sorted
    used_tools = {"Visualization", "Data Profiling", "Code Interpreter"}
    captured_output = io.StringIO()
    sys.stdout = captured_output
    print_used_tools(used_tools)
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    assert "1. Code Interpreter" in output
    assert "2. Data Profiling" in output
    assert "3. Visualization" in output

    print("All print_used_tools tests passed!")


if __name__ == "__main__":
    test_print_used_tools()
