import logging
from transformers import T5ForConditionalGeneration, T5Tokenizer
# Ustawienie poziomu logowania, aby ukryć niepotrzebne komunikaty
logging.getLogger("transformers").setLevel(logging.ERROR)


def zapytaj_t5(polecenie: str) -> str:
    # Załaduj model T5 i tokenizer
    nazwa_modelu = "t5-large"
    model = T5ForConditionalGeneration.from_pretrained(nazwa_modelu)
    tokenizer = T5Tokenizer.from_pretrained(nazwa_modelu)

    # Tokenizacja i kodowanie wejścia
    input_ids = tokenizer.encode(polecenie, return_tensors="pt")

    # Generowanie odpowiedzi
    wyniki = model.generate(input_ids, max_length=50, num_return_sequences=1)
    odpowiedz = tokenizer.decode(wyniki[0], skip_special_tokens=True)

    return odpowiedz


if __name__ == "__main__":
    # Definicja zapytania (poprawna odpowiedź to Fergie)
    zapytanie = "Which member of Black Eyed Peas appeared in Poseidon?"
    # Polecenie bez kontekstu
    polecenie = f"question: {zapytanie}"
    odpowiedz = zapytaj_t5(polecenie=polecenie)
    print(f"Polecenie: {polecenie}")
    print(f"Odpowiedź: {odpowiedz}\n")

    # Polecenie z dodadtkowym kontekstem
    kontekst = """Below are the facts that might be relevant to answer the question: \
(Black Eyed Peas, has part, Fergie), (Black Eyed Peas, has part, Kim Hill), (Poseidon, cast member, Fergie)."""
    polecenie = f"question: {zapytanie} context: {kontekst}"
    odpowiedz = zapytaj_t5(polecenie=polecenie)
    print(f"Polecenie: {polecenie}")
    print(f"Odpowiedź: {odpowiedz}\n")

    # Polecenie z mylącym kontekstem
    kontekst = """Below are the facts that might be relevant to answer the question: \
(Black Eyed Peas, has part, Kim Hill)."""
    polecenie = f"question: {zapytanie} context: {kontekst}"
    odpowiedz = zapytaj_t5(polecenie=polecenie)
    print(f"Polecenie: {polecenie}")
    print(f"Odpowiedź: {odpowiedz}\n")
