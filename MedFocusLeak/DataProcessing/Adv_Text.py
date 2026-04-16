import os
import time
import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# Initialize API
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
client = OpenAI()

def modify_findings(text: str, k: int, retries: int = 3) -> dict:
    """
    Modify a medical findings text by making k small but medically significant changes.
    Returns dict with 'changed_findings'.
    """
    prompt = f"""
You are a clinical findings editor.

Your task is:

1. Make exactly {k} medically significant edits to the original findings.
   - Each edit must be a single word or a very short phrase replacement
     (e.g., 'left' → 'right','mild' → 'moderate','normal' → 'enlarged','stage I' → 'stage II').
   - The edits should be minimal but cause medically significant misleading changes.
   - Do NOT rewrite or rephrase entire sentences—only replace individual terms.
   - Keep the rest of the text identical to the original.

Respond ONLY in the following JSON format:

{{
  "changed_findings": "<findings with {k} edits>"
}}

---

Original findings:
{text}
"""

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert radiologist assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            content = response.choices[0].message.content.strip()
            data = json.loads(content)  # Parse JSON safely
            return data
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2)

    return {"changed_findings": ""}  # fallback if all retries fail


def process_csv(input_csv: str, output_csv: str, k: int):
    # Load from output if it already exists (resume), otherwise from input
    if os.path.exists(output_csv):
        print(f"Resuming from {output_csv}...")
        df = pd.read_csv(output_csv)
    else:
        print(f"Starting fresh from {input_csv}...")
        df = pd.read_csv(input_csv)

    findings_col = f"Findings_{k}"   # dynamic column name

    # Add new column if not exist
    if findings_col not in df.columns:
        df[findings_col] = ""

    # Normalize empties for reliable check
    df[findings_col] = df[findings_col].fillna("").astype(str).str.strip()

    # Filter only rows where findings exist but output is missing
    unprocessed = df[(df["findings"].notna()) & (df[findings_col] == "")]
    print(f"Rows left to process: {len(unprocessed)}")

    for idx, row in tqdm(unprocessed.iterrows(), total=len(unprocessed), desc=f"Processing k={k}"):
        findings = row["findings"]  # Column With the Medical Findings

        result = modify_findings(findings, k)
        df.at[idx, findings_col] = result["changed_findings"]

        # Save progress after each row (auto-resume safe)
        df.to_csv(output_csv, index=False)

    print("Processing complete.")


# Example usage
# process_csv("input.csv", "output.csv", k=5)
