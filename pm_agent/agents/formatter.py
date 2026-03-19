import pandas as pd
import json

class DataFormatterAgent:
    def __init__(self, df: pd.DataFrame, llm_client):
        self.df = df
        self.llm = llm_client

    def run(self) -> pd.DataFrame:
        """
        Analyzes column types using LLM and converts them (int, float, datetime).
        """
        print("Starting Data Formatting...")
        
        # 1. Prepare sample for LLM
        sample = self.df.head(5).to_string()
        if len(sample) > 2000:
            sample = sample[:2000] + "\n... (truncated)"
        columns = list(self.df.columns)
        
        system_prompt = (
            "You are a Data Formatting Agent. Your task is to identify the correct data type for each column based on a sample.\n"
            "Supported types: 'int', 'float', 'datetime', 'string'.\n"
            "Rules:\n"
            "- If a column contains currency or numbers with decimals, use 'float'.\n"
            "- If a column contains whole numbers, use 'int'.\n"
            "- If a column contains dates/times (ISO, diverse formats), use 'datetime'.\n"
            "- Otherwise, use 'string'.\n"
            "Return a JSON object where keys are column names and values are the target types.\n"
            "Example: {\"price\": \"float\", \"date\": \"datetime\", \"user_id\": \"int\"}"
        )
        
        prompt = f"Here is the data sample:\n{sample}\n\nList of columns: {columns}\n\nProvide the types JSON."
        
        response = self.llm.generate_response(prompt, system_prompt)
        type_map = self.llm._parse_json(response)
        
        if not type_map:
            print("Failed to get type map from LLM. Skipping formatting.")
            return self.df

        # 2. Apply conversions
        df_new = self.df.copy()
        for col, target_type in type_map.items():
            if col not in df_new.columns:
                continue
                
            try:
                if target_type == 'int':
                    # Convert to numeric, turn errors to NaN, then fill 0 or similar if needed? 
                    # User said "fix string format", implying clean conversion.
                    # Safe downcast to numeric first
                    df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
                    # Integers cannot contain NaNs in older pandas or standard types, but 'Int64' (nullable) works.
                    # For simplicity, we'll try standard conversion
                    if df_new[col].notna().all():
                         df_new[col] = df_new[col].astype('int64')
                         
                elif target_type == 'float':
                    df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
                    
                elif target_type == 'datetime':
                    df_new[col] = pd.to_datetime(df_new[col], errors='coerce')
                    
                print(f"Column '{col}' converted to {target_type}")
                
            except Exception as e:
                print(f"Error converting column '{col}' to {target_type}: {e}")
                
        return df_new
