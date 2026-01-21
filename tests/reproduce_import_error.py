import pandas as pd
from pm_agent.safe_executor import execute_pandas_code

df = pd.DataFrame({"a": [1, 2]})
code = "import os\nresult = os.name"
res = execute_pandas_code(code, df)
print(f"Result with import: {res}")

code2 = "result = len(df)"
res2 = execute_pandas_code(code2, df)
print(f"Result with len: {res2}")
