from langchain.chains.flare.base import (
    FlareChain,
    QuestionGeneratorChain,
    _OpenAIResponseChain,
)
from langchain.chains.llm import LLMChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import OpenAI
from retriever import SimpleBingRetriever


class MyOpenAIResponseChain(_OpenAIResponseChain):
    """Chain that generates responses from user input and context."""

    def _extract_tokens_and_log_probs(self, generations):
        for gen in generations:
            if gen.generation_info is None:
                raise ValueError

        return (
            gen.generation_info["logprobs"]["tokens"],
            gen.generation_info["logprobs"]["token_logprobs"],
        )


class FLARE:
    def __init__(self, max_iter):
        self.retriever = SimpleBingRetriever()
        self.llm = OpenAI(temperature=0, model_kwargs={"logprobs": 1})
        self.question_generator_chain = QuestionGeneratorChain(llm=self.llm)
        self.response_chain = MyOpenAIResponseChain(llm=self.llm)
        self.flare = FlareChain(
            question_generator_chain=self.question_generator_chain,
            response_chain=self.response_chain,
            retriever=self.retriever,
            min_prob=0.8,
            max_iter=max_iter,
        )
        self.retrieval_chain = RetrievalQA.from_llm(
            retriever=self.retriever, llm=self.llm
        )

    def run(self, query, with_flare=False):
        if with_flare:
            return self.flare.invoke(input=query)
        return self.llm.invoke(input=query)
