import argparse
import time

from freak import FREAK, InputMetadata, RagDocument, RagResult

import httpx


async def call_my_code(_query: str, _metadata: InputMetadata):
    # async with httpx.AsyncClient(timeout=20) as client:
        # response = await client.post(
            # "<YOUR URL HERE>",
            # headers={},
            # json={
                # "query": _query,
                # "metadata": _metadata.dict(),
            # },
        # )
        # response.raise_for_status()
        # response_json = response.json()

    # Turn the response into a RagResult, which is an array of
    # RagDocuments. Each RagDocument is a single document or chunk that
    # will be used to provide RAG context. Here is the full schema:
    return RagResult(
        docs=[
            RagDocument(
                text="<YOUR_DOC_HERE>",
            )
        ],
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--cohere-api-key",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    freak = FREAK(
        comparison_fn=call_my_code,
        cohere_api_key=args.cohere_api_key,
    )
    freak.run(args.verbose)
