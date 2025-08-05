import pytest
from transformers import pipeline

# Import the function you plan to test
from MixtralParaphrasePipeline.paraphraser import paraphrase_spanish_phrase

def test_paraphrase_output_structure():
    input_phrase = "Hola, ¿cómo estás?"
    outputs = paraphrase_spanish_phrase(input_phrase, k=2)

    assert isinstance(outputs, list), "Output should be a list"
    assert len(outputs) == 2, "Should return k paraphrases"
    assert all(isinstance(p, str) for p in outputs), "All items should be strings"
    assert all(len(p.strip()) > 0 for p in outputs), "Paraphrases should not be empty"


