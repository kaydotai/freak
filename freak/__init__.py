import asyncio
import datetime
import inspect
import json
import os
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import cohere
import httpx
import kay
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass
from kay.rag.retrievers import KayRetriever


# KAY_API_URL = "https://api.kay.ai"
KAY_API_URL = "https://embedding-backend-pr-76.onrender.com"


class TermColor:
    HEADER = "\033[95m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    RED = "\033[31m"
    ENDC = "\033[0m"


class DatasetConfig(BaseModel):
    dataset_id: str
    data_types: Optional[List[str]]


class RetrievalConfig(BaseModel):
    instruction: Optional[str]
    num_context: int


class InputMetadata(BaseModel):
    # Metadata for Kay to work with. If you are using a different retrieval
    # API, you can ignore this field.
    dataset_config: DatasetConfig = Field(
        default=DatasetConfig(
            dataset_id="company", data_types=["10-K", "10-Q", "PressRelease"]
        )
    )
    # Metadata for Kay to work with. If you are using a different retrieval
    # API, you can ignore this field. The `num_context` field is the number of
    # documents/chunks we will retrieve.
    retrieval_config: RetrievalConfig = Field(
        default=RetrievalConfig(instruction="Do best retrieval", num_context=3)
    )


class RagDocument(BaseModel):
    text: str
    # Comparands may not have these, so everything below is optional
    company_name: Optional[str] = Field(default=None)
    publish_date: Optional[datetime.datetime] = Field(default=None)
    sic_code_description: Optional[str] = Field(default=None)
    title: Optional[str] = Field(default=None)
    url: Optional[str] = Field(default=None)
    years_mentioned: Optional[List[int]] = Field(default=None)


class RagResult(BaseModel):
    docs: List[RagDocument]
    reasoning_steps: Optional[Any] = Field(default=None)


class EvaluatedRagResult(BaseModel):
    # On top of whatever the internal results are added, we're going to add
    # some more objective measurement. We're not going to do inheritance from
    # the RagResult class, because composition is better ;-)
    rag_result: RagResult
    external_relevancy_scores: List[float]
    external_retrieval_speed: float


@dataclass
class MetricCompare:
    kay_result: float | int
    alternative_result: float | int
    lower_better: bool


@dataclass
class MetadataCompare:
    kay_result: Optional[Any]
    alternative_result: Optional[Any]


ComparisonFn = Callable[[str, InputMetadata], Awaitable[RagResult]]

# A list of questions that we can use to test the comparison function.
DEFAULT_QUESTIONS: List[str] = [
    (
        "Is Johnson & Johnson increasing its marketing budget in 2022?",
        InputMetadata(
            dataset_config=DatasetConfig(
                dataset_id="company",
                data_types=["10-K", "10-Q", "PressRelease"],
            ),
            retrieval_config=RetrievalConfig(
                num_context=3,
                instruction="Do best retrieval",
            ),
        ),
    ),
    (
        "What happened to Intel's acquisition of Tower Semiconductor?",
        InputMetadata(
            dataset_config=DatasetConfig(
                dataset_id="company",
                data_types=["10-K", "10-Q", "PressRelease"],
            ),
            retrieval_config=RetrievalConfig(
                num_context=3,
                instruction="Do best retrieval",
            ),
        ),
    ),
    (
        "What is NVIDIA's earning per share in 2023?",
        InputMetadata(
            dataset_config=DatasetConfig(
                dataset_id="company",
                data_types=["10-K", "10-Q", "PressRelease"],
            ),
            retrieval_config=RetrievalConfig(
                num_context=3,
                instruction="Do best retrieval",
            ),
        ),
    ),
    (
        "What are the main challenges mentioned by T-Mobile in their 2023 annual report?",
        InputMetadata(
            dataset_config=DatasetConfig(
                dataset_id="company",
                data_types=["10-K", "10-Q", "PressRelease"],
            ),
            retrieval_config=RetrievalConfig(
                num_context=3,
                instruction="Do best retrieval",
            ),
        ),
    ),
    (
        "What are the digital transformation initiatives launched by Rolls-Royce Holdings this year?",
        InputMetadata(
            dataset_config=DatasetConfig(
                dataset_id="company",
                data_types=["10-K", "10-Q", "PressRelease"],
            ),
            retrieval_config=RetrievalConfig(
                num_context=3,
                instruction="Do best retrieval",
            ),
        ),
    ),
    (
        "How are different geography segments performing for Apple Inc in 2023 w.r.t net sales?",
        InputMetadata(
            dataset_config=DatasetConfig(
                dataset_id="company",
                data_types=["10-K", "10-Q", "PressRelease"],
            ),
            retrieval_config=RetrievalConfig(
                num_context=3,
                instruction="Do best retrieval",
            ),
        ),
    ),
    (
        "How are user retention numbers for ETSY in 2023?",
        InputMetadata(
            dataset_config=DatasetConfig(
                dataset_id="company",
                data_types=["10-K", "10-Q", "PressRelease"],
            ),
            retrieval_config=RetrievalConfig(
                num_context=3,
                instruction="Do best retrieval",
            ),
        ),
    ),
]


class FREAK:
    __comparison_fn: ComparisonFn
    __cohere_api_key: Optional[str]

    def __init__(
        self,
        *,
        comparison_fn: ComparisonFn,
        cohere_api_key: Optional[str] = None,
    ):
        self.__comparison_fn = comparison_fn
        # TODO custom evaluation fn in case not using cohere
        self.__cohere_api_key = cohere_api_key

    def run(
        self,
        verbose: bool = False,
    ):
        async def call_with_time(
            fn, query: str, metadata: InputMetadata
        ) -> Tuple[RagResult, float]:
            start = time.time()
            if inspect.iscoroutinefunction(fn):
                result = await fn(query, metadata)
            else:
                result = fn(query, metadata)
            end = time.time()
            return result, end - start

        # Read some input (a query + relevant metadata)
        tests = DEFAULT_QUESTIONS

        for query, metadata in tests:
            print(f"{TermColor.BOLD}{TermColor.UNDERLINE}{query}{TermColor.ENDC}")
            kay_result, kay_external_time = asyncio.run(
                call_with_time(self.kay_call, query, metadata)
            )
            alternative_result, alternative_external_time = asyncio.run(
                call_with_time(self.__comparison_fn, query, metadata)
            )

            # Let's add some objective measurement.
            kay_external_relevancy = self._external_relevancy_evaluation(
                query, kay_result.docs
            )
            alternative_external_relevancy = self._external_relevancy_evaluation(
                query, alternative_result.docs
            )
            evaluated_kay_result = EvaluatedRagResult(
                rag_result=kay_result,
                external_relevancy_scores=kay_external_relevancy,
                external_retrieval_speed=kay_external_time,
            )
            evaluated_alternative_result = EvaluatedRagResult(
                rag_result=alternative_result,
                external_relevancy_scores=alternative_external_relevancy,
                external_retrieval_speed=alternative_external_time,
            )

            # Compare the results
            comparison_report = self._compare_results(
                evaluated_kay_result, evaluated_alternative_result
            )

            # Report the results
            self._report_results(
                comparison_report,
                reasoning_steps=kay_result.reasoning_steps if verbose else None,
            )

    def _external_relevancy_evaluation(
        self, query: str, docs: List[RagDocument]
    ) -> List[float]:
        client = cohere.Client(self.__cohere_api_key)
        reranked = client.rerank(
            query=query,
            documents=[doc.text for doc in docs],
            model="rerank-english-v2.0",
        )

        # Return a list of relevance scores that are in the same order as the docs input. Note that
        # the scores in reranked are not in the same order as the docs input.
        doc_to_score = {r.document["text"]: r.relevance_score for r in reranked}
        return [doc_to_score[doc.text] for doc in docs]

    def _compare_results(
        self, kay_result: EvaluatedRagResult, alternative_result: EvaluatedRagResult
    ) -> Dict[str, MetricCompare | MetadataCompare]:
        ret = {}
        # Metric-based comparisons. These are objective numbers that we can measure.
        ret["mean_external_relevancy_score"] = MetricCompare(
            sum(kay_result.external_relevancy_scores)
            / len(kay_result.external_relevancy_scores),
            sum(alternative_result.external_relevancy_scores)
            / len(alternative_result.external_relevancy_scores),
            False,
        )
        ret["max_external_relevancy_score"] = MetricCompare(
            max(kay_result.external_relevancy_scores),
            max(alternative_result.external_relevancy_scores),
            False,
        )
        ret["min_external_relevancy_score"] = MetricCompare(
            min(kay_result.external_relevancy_scores),
            min(alternative_result.external_relevancy_scores),
            False,
        )
        ret["external_retrieval_speed"] = MetricCompare(
            kay_result.external_retrieval_speed,
            alternative_result.external_retrieval_speed,
            True,
        )

        # Metadata-based comparison. These are subjective and binary (metadata exists or not).abs
        ret["has_company_name"] = MetadataCompare(
            kay_result.rag_result.docs[0].company_name,
            alternative_result.rag_result.docs[0].company_name,
        )
        ret["has_publish_date"] = MetadataCompare(
            kay_result.rag_result.docs[0].publish_date,
            alternative_result.rag_result.docs[0].publish_date,
        )
        ret["has_sic_code_description"] = MetadataCompare(
            kay_result.rag_result.docs[0].sic_code_description,
            alternative_result.rag_result.docs[0].sic_code_description,
        )
        ret["has_title"] = MetadataCompare(
            kay_result.rag_result.docs[0].title,
            alternative_result.rag_result.docs[0].title,
        )
        ret["has_url"] = MetadataCompare(
            kay_result.rag_result.docs[0].url,
            alternative_result.rag_result.docs[0].url,
        )
        ret["has_years_mentioned"] = MetadataCompare(
            kay_result.rag_result.docs[0].years_mentioned,
            alternative_result.rag_result.docs[0].years_mentioned,
        )

        return ret

    def _report_results(
        self, comparison_report: Dict[str, Any], reasoning_steps: Optional[Any] = None
    ):
        for test_name, scores_data in comparison_report.items():
            print(f"{TermColor.BOLD}{test_name}{TermColor.ENDC}")
            if type(scores_data) == MetricCompare:
                if scores_data.lower_better:
                    kay_score_color = (
                        TermColor.OKGREEN
                        if scores_data.kay_result < scores_data.alternative_result
                        else TermColor.RED
                    )
                    alternative_score_color = (
                        TermColor.OKGREEN
                        if scores_data.kay_result > scores_data.alternative_result
                        else TermColor.RED
                    )
                else:
                    kay_score_color = (
                        TermColor.OKGREEN
                        if scores_data.kay_result > scores_data.alternative_result
                        else TermColor.RED
                    )
                    alternative_score_color = (
                        TermColor.OKGREEN
                        if scores_data.kay_result < scores_data.alternative_result
                        else TermColor.RED
                    )

                print(
                    f"Kay: {kay_score_color}{scores_data.kay_result}{TermColor.ENDC} Alternative: {alternative_score_color}{scores_data.alternative_result}{TermColor.ENDC}"
                )
            elif type(scores_data) == MetadataCompare:
                if scores_data.kay_result and scores_data.alternative_result:
                    kay_score_color = TermColor.OKGREEN
                    alternative_score_color = TermColor.OKGREEN
                elif not scores_data.kay_result and not scores_data.alternative_result:
                    kay_score_color = TermColor.RED
                    alternative_score_color = TermColor.RED
                elif scores_data.kay_result and not scores_data.alternative_result:
                    kay_score_color = TermColor.OKGREEN
                    alternative_score_color = TermColor.RED
                elif not scores_data.kay_result and scores_data.alternative_result:
                    kay_score_color = TermColor.RED
                    alternative_score_color = TermColor.OKGREEN

                print(
                    f"Kay: {kay_score_color}{scores_data.kay_result}{TermColor.ENDC} Alternative: {alternative_score_color}{scores_data.alternative_result}{TermColor.ENDC}"
                )
            else:
                raise Exception(f"Unknown comparison type: {type(scores_data)}")
        if reasoning_steps is not None:
            reasoning_steps_text = json.dumps(reasoning_steps, indent=2).replace(
                "\\n", "\n"
            )
            print(
                f"{TermColor.BOLD}Reasoning steps that Kay took:{TermColor.ENDC} {reasoning_steps_text}"
            )

    @staticmethod
    async def kay_call(query: str, metadata: InputMetadata) -> RagResult:
        async with httpx.AsyncClient(timeout=20) as client:
            # TODO: we should handle timeouts from both sides
            response = await client.post(
                f"{KAY_API_URL}/retrieve",
                json={
                    "query": query,
                    "dataset_config": metadata.dataset_config.model_dump(mode="json"),
                    "retrieval_config": metadata.retrieval_config.model_dump(
                        mode="json"
                    ),
                    "include_verbose_steps": True,
                },
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Api-Key": "KSbEneM3bs",
                },
            )
            response.raise_for_status()
            response_json = response.json()
            assert len(response_json["contexts"]) == len(response_json["scores"])

        result = RagResult(docs=[])
        for idx, ctx in enumerate(response_json["contexts"]):
            doc = RagDocument(
                text=ctx["chunk_embed_text"],
                company_name=ctx["company_name"],
                publish_date=ctx["data_source_publish_date"],
                sic_code_description=ctx["company_sic_code_description"],
                title=ctx["title"],
                url=ctx["data_source_link"],
                years_mentioned=ctx["chunk_years_mentioned"],
            )
            result.docs.append(doc)
        result.reasoning_steps = response_json["reasoning_steps"]
        return result
