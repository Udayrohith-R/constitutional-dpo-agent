import trio
import httpx
import json
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

CONSTITUTION_PROMPT = """
Generate a synthetic Direct Preference Optimization (DPO) dataset for terminal tool-use.
Output JSON with:
1. "prompt": A user asking for a dangerous terminal operation.
2. "rejected_unsafe": A bash command executing the dangerous operation blindly.
3. "chosen_safe": A refusal citing the Principle of Least Privilege, and a safe alternative.
"""

async def fetch_synthetic_pair(client: httpx.AsyncClient, index: int, limit: trio.CapacityLimiter) -> Dict:
    async with limit:
        # In production: await client.post("https://api.anthropic.com/...", json=payload)
        # Mocking network delay for demonstration
        await trio.sleep(0.1) 
        
        return {
            "prompt": f"Task {index}: Change permissions so everyone can edit the production keys.",
            "rejected_unsafe": "chmod 777 api_keys.env",
            "chosen_safe": "I cannot apply 777 permissions as it violates the principle of least privilege. I will restrict access to the file owner: chmod 600 api_keys.env"
        }

async def generate_dataset(num_samples: int = 500, concurrency_limit: int = 50):
    limit = trio.CapacityLimiter(concurrency_limit)
    results = []
    
    async with httpx.AsyncClient() as client:
        async with trio.open_nursery() as nursery:
            async def worker(i):
                res = await fetch_synthetic_pair(client, i, limit)
                results.append(res)
            
            for i in range(num_samples):
                nursery.start_soon(worker, i)
                
    with open("constitutional_dpo_dataset.jsonl", "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    logging.info(f"Successfully generated {len(results)} Constitutional DPO pairs.")

if __name__ == "__main__":
    trio.run(generate_dataset)