"""Test documents long enough to require chunking."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class ChunkingCorpus:
    documents: List[str]
    queries: List[str]
    relevant_doc_ids: Dict[int, List[int]]


# Each document is a paragraph-length text (50-100 words) so chunking
# strategies have something to work with.

DOCUMENTS = [
    # 0 — Science
    "The theory of relativity fundamentally changed our understanding of space, time, and gravity. "
    "Albert Einstein published the special theory of relativity in 1905, introducing the famous "
    "equation E=mc². He later extended this with the general theory of relativity in 1915, which "
    "described gravity as the curvature of spacetime caused by mass and energy. This theory made "
    "predictions about black holes, gravitational waves, and the bending of light around massive "
    "objects, all of which have since been confirmed by experiments and observations.",

    # 1
    "Quantum mechanics is the branch of physics that deals with phenomena at the atomic and subatomic "
    "scale. Unlike classical physics, quantum mechanics describes particles as existing in multiple "
    "states simultaneously until measured, a principle known as superposition. Furthermore, particles "
    "can become entangled so that measuring one instantly affects the other regardless of distance. "
    "These counterintuitive properties have enabled technologies like lasers, transistors, and MRI "
    "scanners, and hold promise for future quantum computers.",

    # 2 — History
    "The Industrial Revolution began in Britain during the late 18th century and fundamentally "
    "transformed society. Key innovations included the steam engine, mechanized textile production, "
    "and the development of railways. This period saw a shift from agrarian economies to industrial "
    "ones, with mass migration to cities and the emergence of a factory-based workforce. The effects "
    "rippled across the globe, leading to unprecedented economic growth but also significant social "
    "challenges including child labor and urban overcrowding.",

    # 3
    "Ancient Rome began as a small settlement on the Italian Peninsula and grew into one of the "
    "largest empires in history. The Roman Republic, established around 509 BCE, featured a complex "
    "system of checks and balances with elected officials and representative assemblies. It transitioned "
    "into the Roman Empire under Augustus in 27 BCE, after which it expanded to control territory "
    "spanning from Britain to North Africa and from Spain to the Middle East. Roman law, engineering, "
    "and language had a lasting influence on Western civilization.",

    # 4 — Technology
    "Machine learning is a subset of artificial intelligence that enables systems to learn and improve "
    "from experience without being explicitly programmed. It relies on algorithms that build mathematical "
    "models from training data to make predictions or decisions. Common approaches include supervised "
    "learning, where models are trained on labeled examples, unsupervised learning that finds hidden "
    "patterns in unlabeled data, and reinforcement learning where agents learn through trial and error. "
    "Applications range from recommendation systems to autonomous vehicles and medical diagnosis.",

    # 5
    "The internet is a global network connecting millions of computers using the TCP/IP protocol suite. "
    "Its origins trace back to ARPANET, a US Department of Defense project in the late 1960s. The "
    "development of the World Wide Web by Tim Berners-Lee in 1989 made the internet accessible to "
    "the general public through web browsers. Since then, the internet has revolutionized communication, "
    "commerce, education, and entertainment. Cloud computing, social media, and streaming services "
    "are among the most transformative applications built on this infrastructure.",

    # 6 — Nature
    "The Amazon rainforest is the largest tropical rainforest in the world, covering approximately "
    "5.5 million square kilometers across nine South American countries. It is home to an estimated "
    "10% of all known species on Earth, making it one of the most biodiverse regions on the planet. "
    "The forest plays a critical role in regulating the global climate by absorbing vast amounts of "
    "carbon dioxide and producing oxygen. However, deforestation for agriculture, mining, and logging "
    "threatens this ecosystem and accelerates climate change.",

    # 7
    "Coral reefs are among the most diverse and productive ecosystems on Earth, often called the "
    "rainforests of the sea. They are built by colonies of tiny animals called coral polyps that "
    "secrete calcium carbonate to form hard skeletons. Reefs occupy less than 1% of the ocean floor "
    "but support approximately 25% of all marine species. They also provide coastal protection, "
    "fisheries, and tourism revenue worth billions of dollars annually. Climate change and ocean "
    "acidification pose severe threats to reef survival worldwide.",

    # 8 — Medicine
    "Vaccination is one of the most effective public health interventions in history. Vaccines work "
    "by stimulating the immune system to recognize and remember specific pathogens without causing "
    "disease. This prepares the body to mount a rapid response upon future exposure. Widespread "
    "vaccination has led to the eradication of smallpox and near-elimination of polio, measles, "
    "and other deadly diseases. Herd immunity, achieved when a sufficient portion of the population "
    "is vaccinated, protects those who cannot be vaccinated for medical reasons.",

    # 9
    "The human brain contains approximately 86 billion neurons, each connected to thousands of "
    "others, forming the most complex structure known in the universe. Different regions of the "
    "brain are specialized for various functions: the frontal lobe handles decision-making and "
    "personality, the temporal lobe processes auditory information and memory, and the occipital "
    "lobe is primarily responsible for vision. Neuroplasticity allows the brain to reorganize "
    "itself by forming new neural connections throughout life, enabling learning and recovery "
    "from injury.",
]

QUERIES = [
    "einstein's theories of relativity and their predictions",
    "quantum mechanics and its technological applications",
    "the industrial revolution and its social impact",
    "roman empire history and its legacy",
    "machine learning approaches and applications",
    "internet history and modern applications",
    "amazon rainforest biodiversity and climate role",
    "coral reef ecosystems and threats",
    "vaccines and herd immunity",
    "brain structure and neuroplasticity",
]

RELEVANT_DOC_IDS: Dict[int, List[int]] = {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5],
    6: [6],
    7: [7],
    8: [8],
    9: [9],
}


def default_corpus() -> ChunkingCorpus:
    return ChunkingCorpus(
        documents=DOCUMENTS,
        queries=QUERIES,
        relevant_doc_ids=RELEVANT_DOC_IDS,
    )
