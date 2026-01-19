import json
import re

def _parse_llm_json(response_str: str) -> dict:
    """Robustly parse JSON from LLM response."""
    if not response_str:
        return {}

    cleaned_str = response_str.strip()
    
    # 1. Simple try
    try:
        return json.loads(cleaned_str)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown blocks
    if "```json" in cleaned_str:
        try:
            # Extract content between ```json and ```
            json_block = cleaned_str.split("```json")[1].split("```")[0].strip()
            return json.loads(json_block)
        except (IndexError, json.JSONDecodeError):
            pass
    elif "```" in cleaned_str:
        # Try generic ``` blocks
        try:
            parts = cleaned_str.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("{") or part.startswith("["):
                    try:
                        return json.loads(part)
                    except json.JSONDecodeError:
                        continue
        except IndexError:
            pass

    # 3. Regex search for first { or [ to last } or ]
    json_match = re.search(r"(\{.*\}|\[.*\])", cleaned_str, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except (json.JSONDecodeError, ValueError):
            pass

    return None

def test_parsing():
    cases = [
        # Pure JSON
        '{"key": "value"}',
        # Markdown JSON
        '```json\n{"key": "value"}\n```',
        # Conversational + JSON
        'Here is the result:\n{"key": "value"}\nHope it helps!',
        # Multiple code blocks, one is JSON
        'Explanation...\n```python\nprint(1)\n```\nAnd code:\n```json\n{"key": "value"}\n```',
        # Malformed JSON with text around
        'The answer is {"key": "value"} definitely.',
        # Nested JSON structure in text
        'Complex response: { "nested": {"a": 1}, "list": [1,2,3] } end.',
    ]

    for i, case in enumerate(cases):
        result = _parse_llm_json(case)
        print(f"Case {i}: {'SUCCESS' if result and result.get('key') == 'value' or (i==5 and result and 'nested' in result) else 'FAIL'}")
        if not result:
            print(f"  Result was None for: {case[:50]}...")

if __name__ == "__main__":
    test_parsing()
