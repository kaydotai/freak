FREAK
=

*F*inance *R*AG *E*valuation for *A*ssessing *K*ay

FREAK aims to provide an evaluation framework for running analyses against Kay's finance data RAG. Plug in your API call and call `run` — we'll take care of the rest.

## Quick Start

1.  In `tester.py` fill out the `call_my_code` function with a call to your API. Convert the response into a `RagResult`. (more details below)
2. Get a `cohere` API key (if you don't have one, you can create one here - [cohere signup](https://cohere.com/))
3. Run  `python tester.py [--verbose] --cohere-api-key <YOUR_API_KEY_HERE>`

If you want to add custom queries, add them `__init__.py` `DEFAULT_QUESTIONS`

## Detailed Usage

### Details on `RagResult` 
`RagResult` helps us bring retrived results to a common structure for easy comparision. 

`RagResult` is an array of `RagDocument`. 
```python
RagResult(
        docs=[
            RagDocument
        ],
    )
```
Full defination is here - [RagResult](https://github.com/kaydotai/freak/blob/main/freak/__init__.py#L71)

### Details on `RagDocument`
`RagDocument` holds the raw text of the retrieved context, along with optional metadata.

```python
doc1 = RagDocument(text = "<TEXT_OF_YOUR_DOC_HERE>")
```

Full defination is here - [RagResult](https://github.com/kaydotai/freak/blob/main/freak/__init__.py#L60)


## Why do I need a cohere API key?
We use cohere re-ranking scores as a proxy to evaluate relevancy for retrieved context. While we acknowledge the evident short-comings, this is a quick way to do sanity checks between two retriever systems without a golden test set. If you have a golden test set internally, we can add more metrics to compare two retriver systems. 


## Contribution
At Kay, we are pushing the boundaries on RAG. One of the biggest challenges is to accurately keep evaluating a retriever system. The intention behind this library is twofold - 
1. We use this internally to test improvements confidently and track changes
2. We made this publicly available to enable our users to test Kay's retriever system with theirs.

With that note, we would love for you to contribute to this package.
