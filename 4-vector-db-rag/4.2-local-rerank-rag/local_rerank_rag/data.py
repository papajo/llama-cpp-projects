"""Built-in test corpus with documents, queries, and relevance judgments."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class RagCorpus:
    documents: List[str]
    queries: List[str]
    relevant_doc_ids: Dict[int, List[int]]


DEFAULT_DOCUMENTS = [
    # Science
    "The theory of relativity was developed by Albert Einstein in 1905 and 1915.",
    "Quantum computing uses qubits that can exist in superposition states.",
    "CRISPR-Cas9 is a gene-editing tool derived from bacterial immune systems.",
    "The human genome contains approximately 3 billion base pairs of DNA.",
    "Neutrinos are nearly massless subatomic particles that rarely interact with matter.",
    # History
    "The printing press was invented by Johannes Gutenberg around 1440 in Europe.",
    "The Cold War was a period of geopolitical tension between the US and Soviet Union.",
    "The Berlin Wall fell in 1989, leading to German reunification in 1990.",
    "Ancient Greece is considered the birthplace of Western democracy and philosophy.",
    "The Silk Road was an ancient network of trade routes connecting East and West.",
    # Technology
    "Python is a high-level programming language known for its readability and ecosystem.",
    "Docker containers package software with its dependencies for consistent deployment.",
    "Git is a distributed version control system created by Linus Torvalds in 2005.",
    "PostgreSQL is a powerful open-source relational database management system.",
    "REST APIs use HTTP methods to perform CRUD operations on web resources.",
    # Nature
    "The Great Barrier Reef is the world's largest coral reef system off Australia.",
    "Migration patterns of monarch butterflies span multiple generations across North America.",
    "Mycorrhizal networks connect tree roots underground, enabling nutrient exchange.",
    "The Mariana Trench is the deepest oceanic trench on Earth at about 11 km.",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
    # Medicine
    "Vaccines work by exposing the immune system to a harmless part of a pathogen.",
    "Antibiotics target bacterial cell walls or protein synthesis to kill infections.",
    "MRI scanners use strong magnetic fields to produce detailed images of organs.",
    "The placebo effect describes real physiological changes from belief in treatment.",
    "Telomeres are protective caps at chromosome ends that shorten with cell division.",
]

DEFAULT_QUERIES = [
    "genetic engineering and dna editing",
    "cold war history and political tension",
    "software development tools and version control",
    "ocean ecosystems and marine biology",
    "how vaccines and antibiotics fight disease",
    "ancient trade routes and cultural exchange",
    "quantum physics and subatomic particles",
    "container technology and software deployment",
    "coral reefs and ocean ecosystems",
    "medical imaging and diagnostic technology",
]

DEFAULT_RELEVANT_DOC_IDS: Dict[int, List[int]] = {
    0: [2, 3],       # genetic engineering / dna editing
    1: [6, 7],       # cold war / political tension
    2: [10, 11, 12], # software dev / version control
    3: [15, 16, 18], # ocean / marine
    4: [20, 21, 24], # vaccines / antibiotics
    5: [9],          # trade routes
    6: [1, 4],       # quantum / subatomic
    7: [11],         # containers / deployment
    8: [15, 17],     # coral reefs / ecosystems
    9: [22, 23],     # medical imaging
}


def default_corpus() -> RagCorpus:
    return RagCorpus(
        documents=DEFAULT_DOCUMENTS,
        queries=DEFAULT_QUERIES,
        relevant_doc_ids=DEFAULT_RELEVANT_DOC_IDS,
    )


def mini_corpus() -> RagCorpus:
    return RagCorpus(
        documents=[
            "Einstein developed the theory of relativity.",
            "CRISPR is a gene-editing tool from bacterial immune systems.",
            "The Cold War was tension between US and Soviet Union.",
            "Docker packages software for consistent deployment.",
            "The Great Barrier Reef is off the coast of Australia.",
        ],
        queries=[
            "gene editing and genetics",
            "international political conflict",
        ],
        relevant_doc_ids={
            0: [1],
            1: [2],
        },
    )
