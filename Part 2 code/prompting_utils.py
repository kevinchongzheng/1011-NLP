import os
import re


def read_schema(schema_path):
    '''
    Read the .schema file and return its contents.
    
    Args:
        schema_path (str): Path to the schema file
    
    Returns:
        str: The schema content, or empty string if file doesn't exist
    '''
    if not os.path.exists(schema_path):
        print(f"Warning: Schema file not found at {schema_path}")
        return ""
    
    try:
        with open(schema_path, 'r') as f:
            schema_text = f.read()
        return schema_text
    except Exception as e:
        print(f"Error reading schema file: {e}")
        return ""


def extract_sql_query(response):
    '''
    Extract the SQL query from the model's response.
    
    The model might output SQL in various formats:
    - Plain SQL
    - SQL wrapped in markdown code blocks (```sql ... ```)
    - SQL with additional explanatory text
    
    Args:
        response (str): The full response from the model
    
    Returns:
        str: The extracted SQL query, stripped of whitespace
    '''
    # Try to find SQL in markdown code blocks first
    sql_block_pattern = r'```sql\s*(.*?)\s*```'
    matches = re.findall(sql_block_pattern, response, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip()
    
    # Try generic code blocks
    code_block_pattern = r'```\s*(.*?)\s*```'
    matches = re.findall(code_block_pattern, response, re.DOTALL)
    if matches:
        # Return the first code block that looks like SQL
        for match in matches:
            if 'SELECT' in match.upper():
                return match.strip()
    
    # Look for SQL keywords to find where SQL might start
    # Common pattern: "SQL: SELECT ..." or just "SELECT ..."
    sql_start_pattern = r'(?:SQL\s*:\s*)?(SELECT.*?)(?:\n\n|$)'
    matches = re.findall(sql_start_pattern, response, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip()
    
    # If no clear SQL found, try to find anything starting with SELECT
    select_pattern = r'(SELECT\s+.*?)(?:\n\n|$|```)'
    matches = re.findall(select_pattern, response, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip()
    
    # Last resort: return the whole response stripped
    # (might contain extra text, but better than nothing)
    return response.strip()


def save_logs(output_path, sql_em, record_em, record_f1, error_msgs):
    '''
    Save the logs of the experiment to files.
    
    Args:
        output_path (str): Path to save the log file
        sql_em (float): SQL exact match score
        record_em (float): Record exact match score
        record_f1 (float): Record F1 score
        error_msgs (list): List of error messages from SQL execution
    '''
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(f"SQL Exact Match: {sql_em}\n")
        f.write(f"Record Exact Match: {record_em}\n")
        f.write(f"Record F1: {record_f1}\n")
        f.write(f"\nNumber of errors: {sum(1 for msg in error_msgs if msg)}\n")
        f.write(f"Total queries: {len(error_msgs)}\n")
        f.write(f"\nError Messages:\n")
        for i, msg in enumerate(error_msgs):
            if msg:
                f.write(f"Query {i}: {msg}\n")
