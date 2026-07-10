"""Built-in test corpus with relevance judgments for ablation experiments."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class Corpus:
    """A collection of documents and queries with relevance judgments."""

    documents: List[str]
    queries: List[str]
    relevant_doc_ids: Dict[str, List[int]] = field(default_factory=dict)
    # Maps query_index → list of relevant document indices


# ---------------------------------------------------------------------------
# Default mini-corpus — 30 documents across 5 topics, 10 queries
# ---------------------------------------------------------------------------

DEFAULT_DOCUMENTS = [
    # Science — 0–5
    "The theory of relativity was developed by Albert Einstein in the early 20th century.",
    "Quantum mechanics describes the behavior of particles at the atomic and subatomic scale.",
    "DNA, or deoxyribonucleic acid, carries the genetic instructions used in living organisms.",
    "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
    "The periodic table organizes chemical elements by their atomic number and properties.",
    "Evolution by natural selection was first described by Charles Darwin in 1859.",
    # History — 6–11
    "The Roman Empire was one of the largest empires in ancient history, lasting over 500 years.",
    "World War II was a global war that lasted from 1939 to 1945 involving many nations.",
    "The French Revolution began in 1789 and radically transformed French society and government.",
    "Ancient Egypt developed along the Nile River and is known for its pyramids and pharaohs.",
    "The Industrial Revolution began in Britain in the late 18th century and spread worldwide.",
    "The United Nations was founded in 1945 after World War II to promote international cooperation.",
    # Technology — 12–17
    "Machine learning is a subset of artificial intelligence focused on pattern recognition and prediction.",
    "The internet is a global network of interconnected computers that communicate via TCP/IP.",
    "Encryption is the process of encoding information so that only authorized parties can access it.",
    "Blockchain is a distributed ledger technology that underpins cryptocurrencies like Bitcoin.",
    "Cloud computing delivers computing services over the internet on a pay-as-you-go basis.",
    "The transistor, invented in 1947, is the fundamental building block of modern electronic devices.",
    # Nature — 18–23
    "The Amazon rainforest is the largest tropical rainforest in the world, spanning nine countries.",
    "Coral reefs are diverse underwater ecosystems held together by calcium carbonate structures.",
    "Migration is the regular seasonal movement of animals from one region to another.",
    "The water cycle describes the continuous movement of water through evaporation, condensation, and precipitation.",
    "Tectonic plates are large slabs of Earth's lithosphere that move and cause earthquakes.",
    "Biodiversity refers to the variety of life forms in a given habitat or ecosystem.",
    # Sports — 24–29
    "The Olympic Games are a major international sporting event held every four years.",
    "Football, known as soccer in some countries, is the world's most popular sport by participation.",
    "Chess is a two-player strategy board game with origins dating back over 1,500 years.",
    "Marathon running is a long-distance foot race with a standard distance of 42.195 kilometres.",
    "The World Cup is the most prestigious association football tournament in the world.",
    "Yoga is an ancient practice originating in India that combines physical postures and breathing.",
]

DEFAULT_QUERIES = [
    "scientific theories about the universe",
    "world war history and international organizations",
    "computer systems and digital technology",
    "natural environments and ecosystems",
    "global sporting competitions",
    "biology of living organisms",
    "ancient civilizations and empires",
    "data security and cryptography",
    "earth science and geological processes",
    "artificial intelligence and computing",
]

DEFAULT_RELEVANT_DOC_IDS: Dict[int, List[int]] = {
    # "scientific theories about the universe"
    0: [0, 1],
    # "world war history and international organizations"
    1: [7, 11],
    # "computer systems and digital technology"
    2: [13, 14, 16],
    # "natural environments and ecosystems"
    3: [18, 19, 21, 23],
    # "global sporting competitions"
    4: [24, 25, 27, 28],
    # "biology of living organisms"
    5: [2, 3, 5],
    # "ancient civilizations and empires"
    6: [6, 8, 9],
    # "data security and cryptography"
    7: [14],
    # "earth science and geological processes"
    8: [21, 22, 24],
    # "artificial intelligence and computing"
    9: [12, 15, 17],
}


def default_corpus() -> Corpus:
    """Return the default test corpus with relevance judgments."""
    return Corpus(
        documents=DEFAULT_DOCUMENTS,
        queries=DEFAULT_QUERIES,
        relevant_doc_ids=DEFAULT_RELEVANT_DOC_IDS,
    )


def mini_corpus() -> Corpus:
    """Return a tiny 5-document, 2-query corpus for quick tests."""
    return Corpus(
        documents=[
            "Albert Einstein developed the theory of relativity.",
            "Quantum mechanics describes subatomic particle behavior.",
            "The Amazon is the world's largest tropical rainforest.",
            "Coral reefs are diverse underwater ecosystems.",
            "The Olympic Games are held every four years.",
        ],
        queries=[
            "physics and scientific theories",
            "natural ecosystems",
        ],
        relevant_doc_ids={
            0: [0, 1],
            1: [2, 3],
        },
    )
