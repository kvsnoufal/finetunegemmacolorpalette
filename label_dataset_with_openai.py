

with open("./data/palettes.json","r") as f:
    loaded = f.read()

loaded = eval(loaded)
print(len(loaded))
import pandas as pd

df = pd.DataFrame(loaded)
print(df.shape)
print(df.head())
from litellm import completion
import os
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel
import json

proxy_base = os.getenv("OPENAI_API_BASE")
proxy_token = os.getenv("OPENAI_API_KEY")
model = os.getenv("WRITER_MODEL",    "openai/gpt-4o")
SP = '''
Give a brief one liner phrase about the color palette. Maximum 6 words:
'''

from pydantic import BaseModel,Field
from litellm import completion
from tqdm import tqdm
class ResponseModel(BaseModel):
    response: str = Field(...)
    class Config:
        extra = "forbid"

import json
schema_generator = ResponseModel.model_json_schema()

import concurrent.futures
import pandas as pd
import json
from tqdm import tqdm

# Parameters
SAVE_EVERY = 20  # Save after every 20 rows
MAX_WORKERS = 5  # Number of parallel workers

def process_row(idx, row):
    cs = row["colors"]
    messages = [
        {"role": "system", "content": SP},
        {"role": "user", "content": f"Palette :\n{str(cs)}"}
    ]
    try:
        resp = completion(
            messages=messages,
            api_base=proxy_base,
            api_key=proxy_token,
            model=model,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response_schema",
                    "schema": schema_generator,
                    "strict": True,
                },
            },
        )
        content = resp["choices"][0]["message"]["content"]
        struct_output = json.loads(content)["response"]
        return idx, struct_output
    except:
        return idx, None

# Parallel Execution
results = [None] * len(df)
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_row, i, row): i for i, row in df.iterrows()}

    for count, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(futures))):
        i, output = future.result()
        results[i] = output

        # Save output to df as we go
        if output is not None:
            df.loc[i, 'resp'] = output

        if count % SAVE_EVERY == 0:  # Periodically save
            df.to_csv("./data/par_dfdata.csv", index=None)

# Final Save after all tasks complete
df.to_csv("./data/par_dfdata.csv", index=None)
