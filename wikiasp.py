import sys
import os
import logging
import torch
from datasets import load_dataset
from langchain.chains.flare.base import (
    FlareChain,
    QuestionGeneratorChain,
    _OpenAIResponseChain,
)
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import OpenAI
from retriever import SimpleBingRetriever
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score


# Suppress logging
logging.getLogger().setLevel(logging.CRITICAL)

# Add unieval folder to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "unieval")))
from utils import convert_to_json
from metric.evaluator import get_evaluator


class SimpleCombineDocumentsChain(BaseCombineDocumentsChain):
    def combine_docs(self, docs):
        combined_text = "\n\n".join(doc.page_content for doc in docs)
        return combined_text

    async def acombine_docs(self, docs):
        return self.combine_docs(docs)


class MyOpenAIResponseChain(_OpenAIResponseChain):
    def _extract_tokens_and_log_probs(self, generations):
        for gen in generations:
            if gen.generation_info is None:
                raise ValueError
        return (
            gen.generation_info["logprobs"]["tokens"],
            gen.generation_info["logprobs"]["token_logprobs"],
        )


class FLARE:
    def __init__(self, max_iter, file_path="wikiasp/data-00000-of-00001.arrow"):
        self.file_path = file_path
        self.dataset = self.load_data()

        # Initialize FLARE components
        self.retriever = SimpleBingRetriever()
        self.llm = OpenAI(temperature=0, model_kwargs={"logprobs": 1})
        self.question_generator_chain = QuestionGeneratorChain(llm=self.llm)
        self.response_chain = MyOpenAIResponseChain(llm=self.llm)
        self.flare = FlareChain(
            question_generator_chain=self.question_generator_chain,
            response_chain=self.response_chain,
            retriever=self.retriever,
            min_prob=0.6,
            max_iter=max_iter,
        )
        self.retrieval_chain = RetrievalQA.from_llm(
            retriever=self.retriever, llm=self.llm
        )

    def load_data(self):
        dataset = load_dataset("arrow", data_files=self.file_path)
        return dataset["train"].select_columns(
            ["clean_targets", "clean_title", "domain"]
        )

    def get_aspects(self, sample):
        return [aspect for aspect, _ in sample["clean_targets"]]

    def generate_summary(self, sample):
        summary_lines = []
        for aspect, description in sample["clean_targets"]:
            summary_lines.append(f"{aspect}: {description}")
        return "\n".join(summary_lines)

    def run_experiments(self, n=1, with_flare=False):
        if self.dataset is not None:
            rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            rouge_scores = []
            e_f1_scores = []
            unieval_scores = []

            for i in range(min(n, len(self.dataset))):
                sample = self.dataset[i]
                aspects = self.get_aspects(sample)
                summary = self.generate_summary(sample)
                aspects_str = ", ".join(aspects)

                few_shot_example = """
The following example helps to determine the format of the output:
Generate a summary about Aslanhane Mosque including the following aspects: location, history with one aspect per line.
Location: The mosque is in the old quarter of ankara next to ankara castle. With an altitude of 947 metres (3,107 ft) it overlooks ankara
at 39°56’12"N 32°51’55"E.
History: The mosque is one of the oldest mosques in Turkey still standing. It was built during the reign of Mesud II of the Anatolian
Seljuks in 1290. Its architect was Ebubekir Mehmet.
"""

                question = (
                    f"{few_shot_example}\n"
                    f"Generate a summary about '{sample['clean_title']}' including the following aspects: {aspects_str}."
                )

                print(f"Sample {i + 1} Aspects: {aspects}\n")
                print(f"Sample {i + 1} Gold Answer:\n{summary}\n")
                print(f"Sample {i + 1} Question:\n{question}\n")

                response = self.run(query=question, with_flare=with_flare)
                answer = response["response"] if "response" in response else response
                print(f"Sample {i + 1} Answer:\n{answer}\n")

                rouge_score = rouge.score(summary, answer)["rougeL"].fmeasure
                rouge_scores.append(rouge_score)

                gold_entities = set(summary.split())
                answer_entities = set(answer.split())
                true_positive = len(gold_entities & answer_entities)
                precision = (
                    true_positive / len(answer_entities) if answer_entities else 0
                )
                recall = true_positive / len(gold_entities) if gold_entities else 0
                e_f1_score = (
                    (2 * precision * recall) / (precision + recall)
                    if (precision + recall)
                    else 0
                )
                e_f1_scores.append(e_f1_score)

                data = convert_to_json(
                    output_list=[answer], src_list=[question], ref_list=[summary]
                )c
                evaluator = get_evaluator("summarization", device="cpu")
                eval_scores = evaluator.evaluate(data)

                # Print the scores
                print(f"Coherence: {eval_scores[0]['coherence']:.2f}")
                print(f"Consistency: {eval_scores[0]['consistency']:.2f}")
                print(f"Fluency: {eval_scores[0]['fluency']:.2f}")
                print(f"Relevance: {eval_scores[0]['relevance']:.2f}")
                print(f"Overall: {eval_scores[0]['overall']:.2f}")

                unieval_scores.append(eval_scores[0]["overall"])

                print(f"Sample {i + 1} Rouge-L Score: {rouge_score:.2f}")
                print(f"Sample {i + 1} E-F1 Score: {e_f1_score:.2f}")
                print(f"Sample {i + 1} UniEval Score: {eval_scores[0]['overall']:.2f}")

                print("\n" + "-" * 80 + "\n")

            print(
                f"Average Rouge-L Score for {n} samples: {(sum(rouge_scores) / n):.2f}"
            )
            print(f"Average E-F1 Score for {n} samples: {(sum(e_f1_scores) / n):.2f}")
            print(
                f"Average UniEval Score for {n} samples: {(sum(unieval_scores) / n):.2f}"
            )

        else:
            raise ValueError("Dataset not loaded. Please call load_data() first.")

    def run(self, query, with_flare=False):
        if with_flare:
            return self.flare.invoke(input=query)
        return self.llm.invoke(input=query)


if __name__ == "__main__":
    flare = FLARE(max_iter=4)
    flare.run_experiments(n=1, with_flare=False)
