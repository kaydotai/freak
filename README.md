FREAK
=

*F*inance *R*AG *E*valuation for *A*ssessing *K*ay

FREAK aims to provide an evaluation framework for running analyses against Kay's finance data RAG. Plug in your API call and call `run` — we'll take care of the rest.

## Usage

1) In `tester.py` fill out the `call_my_code` function with a call to your API, and convert the response into a `RagDocument`.
2) Get a `cohere` API key
3) Run something like `python tester.py [--verbose] --cohere-api-key <YOUR_API_KEY_HERE>`

If you want to add custom queries, add them `__init__.py` `DEFAULT_QUESTIONS`