import unittest
from pm_agent.safe_executor import validate_code_syntax

class TestCodeValidation(unittest.TestCase):
    def test_valid_code(self):
        code = "result = df.groupby('case_id').size()"
        res = validate_code_syntax(code)
        self.assertTrue(res["success"])

    def test_syntax_error(self):
        code = "result = df.groupby('case_id').size("  # Missing parenthesis
        res = validate_code_syntax(code)
        self.assertFalse(res["success"])
        self.assertIn("Синтаксическая ошибка", res["error"])

    def test_complex_valid_code(self):
        code = """
import pandas as pd
def solve():
    return df.head()
result = solve()
"""
        res = validate_code_syntax(code)
        self.assertTrue(res["success"])

    def test_indentation_error(self):
        code = """
def solve():
return df.head()
"""
        res = validate_code_syntax(code)
        self.assertFalse(res["success"])
        self.assertIn("Синтаксическая ошибка", res["error"])

if __name__ == "__main__":
    unittest.main()
