import re
import os
import json
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from wikidata_retriever import WikidataRetriever

os.environ["TOKENIZERS_PARALLELISM"] = "false"

FEWSHOT_EXAMPLES = """Question: What is the seventh tallest mountain in North America?
Example Output: Mount Lucania
Question: What year was the first book of the A Song of Ice and Fire series published?
Example Output: 1996
Question: How old was Taylor Swift when she won her first Grammy?
Example Output: 20
Question: Has there ever been a Christian U.S. senator?
Example Output: Yes"""

INSTRUCTIONS = """Answer the user's question as concisely as possible.
If the answer is a number, then response should be just number, not text."""

RESULTS_DIR = "results"
EXPERIMENTS = 40


class KAPINGChain:
    def __init__(
        self,
        with_kaping: bool = True,
        with_caching: bool = True,
        top_k: int = 3,
    ):
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        self.with_kaping = with_kaping
        self.with_caching = with_caching
        self.llm = OpenAI(temperature=0)
        self.top_k = top_k
        self.retriever = WikidataRetriever(with_caching=with_caching)
        self.__kaping_prompt_template = PromptTemplate(
            input_variables=["question_number", "question", "fewshots", "entities"],
            template=f"""{INSTRUCTIONS}
Here are some few-shot examples for output format:
{{fewshots}}
Below are the facts that might be relevant to answer the question, but they may not:
{{entities}}
Question {{question_number}}: {{question}}
Answer:""",
        )
        self.__default_prompt_template = PromptTemplate(
            input_variables=["question_number", "question", "fewshots"],
            template=f"""{INSTRUCTIONS}
Here are some few-shot examples for output format:
{{fewshots}}
Question {{question_number}}: {{question}}
Answer:""",
        )

    def __save_cache(self, filename: str, data: dict) -> None:
        if self.with_caching:
            with open(os.path.join(RESULTS_DIR, filename), "w") as f:
                json.dump(data, f)

    def __load_cache(self, filename: str) -> dict:
        cache_file = os.path.join(RESULTS_DIR, filename)
        if self.with_caching and os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                return json.load(f)

    def answer_question(
        self, question_number: int, question: str, entities: list, reference: str
    ) -> str:
        cache_filename = f"result_q_{question_number}_k_{self.top_k}_kaping_{self.with_kaping}.json"
        if cached_result := self.__load_cache(cache_filename):
            print(
                f"Loaded cached result for question {question_number} while with_kaping={self.with_kaping}"
            )
            return cached_result["response"]

        if self.with_kaping:
            top_entities = self.retriever.top_k_neighbors(
                question, entities, self.top_k
            )
            str_top_entities = ", ".join(
                [f"({', '.join(entity)})" for entity in top_entities]
            )
            prompt = self.__kaping_prompt_template.format(
                question_number=question_number,
                question=question,
                fewshots=FEWSHOT_EXAMPLES,
                entities=str_top_entities,
            )
        else:
            prompt = self.__default_prompt_template.format(
                question_number=question_number,
                question=question,
                fewshots=FEWSHOT_EXAMPLES,
            )

        response = self.llm.invoke(prompt)
        cleaned_response = self.__clean_text(response)
        print(f"Question: {question}")
        print(f"Generated Response: {cleaned_response}")
        print(f"Expected Answer: {reference}")

        self.__save_cache(
            cache_filename, {"response": cleaned_response, "reference": reference}
        )

        return cleaned_response

    def __clean_text(self, text: str) -> str:
        return re.sub(r"\W+", " ", text).strip()

    def evaluate(self, predictions: list, references: list) -> dict:
        cleaned_predictions = list(map(self.__clean_text, predictions))
        cleaned_references = list(map(self.__clean_text, references))

        precision, recall, f1, _ = precision_recall_fscore_support(
            cleaned_references, cleaned_predictions, average="macro", zero_division=0
        )
        accuracy = accuracy_score(cleaned_references, cleaned_predictions)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }

    def run_experiments(self, experiments=range(EXPERIMENTS)) -> dict:
        selected_dataset = self.retriever.dataset["test"].select(experiments)

        predictions = [
            self.answer_question(
                i + 1, entry["question"], entry["questionEntity"], entry["answerText"]
            )
            for i, entry in enumerate(selected_dataset)
        ]

        results = self.evaluate(
            predictions, [entry["answerText"] for entry in selected_dataset]
        )

        print("Results:")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1']:.4f}")
        print(f"Accuracy: {results['accuracy']:.4f}")

        self.__save_cache(
            f"evaluation_k_{self.top_k}_kaping_{self.with_kaping}_exp_{len(experiments)}.json",
            results,
        )
        return results

    def compare(
        self,
        other_chain: "KAPINGChain",
        experiments: range = range(EXPERIMENTS),
    ) -> dict:
        results_with_kaping = self.__load_cache(
            f"evaluation_k_{self.top_k}_kaping_{self.with_kaping}_exp_{len(experiments)}.json"
        ) or self.run_experiments(experiments)

        results_without_kaping = other_chain.__load_cache(
            f"evaluation_k_{self.top_k}_kaping_{other_chain.with_kaping}_exp_{len(experiments)}.json"
        ) or other_chain.run_experiments(experiments)

        print("Comparison of Results:")
        print(f"{'Metric':<15}{'With KAPING':<15}{'Without KAPING'}")
        print(
            f"{'Accuracy':<15}{results_with_kaping['accuracy']:<15.4f}{results_without_kaping['accuracy']:.4f}"
        )
        print(
            f"{'Precision':<15}{results_with_kaping['precision']:<15.4f}{results_without_kaping['precision']:.4f}"
        )
        print(
            f"{'Recall':<15}{results_with_kaping['recall']:<15.4f}{results_without_kaping['recall']:.4f}"
        )
        print(
            f"{'F1 Score':<15}{results_with_kaping['f1']:<15.4f}{results_without_kaping['f1']:.4f}"
        )

        return {
            "with_kaping": results_with_kaping,
            "without_kaping": results_without_kaping,
        }


if __name__ == "__main__":
    with_kaping = KAPINGChain()
    without_kaping = KAPINGChain(with_kaping=False)
    with_kaping.compare(without_kaping)
