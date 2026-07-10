"""
Pre-built chain templates for common patterns.

Each function returns a ``PromptChain`` object ready to use.
Templates available::

    from prompt_chaining.templates import (
        summarize_and_translate,
        brainstorm_critique_refine,
        expand_and_polish,
        question_decomposition,
        fact_check_pipeline,
        chain_of_thought,
    )

    chain = summarize_and_translate()
    chain2 = brainstorm_critique_refine()
"""

from __future__ import annotations

from .chain import ChainStep, PromptChain


def summarize_and_translate() -> PromptChain:
    """Summarise input text, then translate to French.

    Steps:
        1. summarise — Condense the input to key points.
        2. translate — Translate the summary to French.
    """
    return PromptChain(
        name="summarize-and-translate",
        description="Summarise input text, then translate the summary to French.",
        temperature=0.3,
        steps=[
            ChainStep(
                name="summarise",
                system_prompt="You are a precise summariser. Extract the key points concisely.",
                user_prompt="Summarise the following:\\n\\n{{input}}",
                temperature=0.2,
            ),
            ChainStep(
                name="translate",
                system_prompt="You are a professional translator. Translate accurately.",
                user_prompt="Translate this to French:\\n\\n{{step_1}}",
            ),
        ],
    )


def brainstorm_critique_refine() -> PromptChain:
    """Generate ideas, critique them, then refine.

    Steps:
        1. brainstorm — Generate creative ideas on a topic.
        2. critique — Critically analyse the ideas for flaws.
        3. refine — Produce an improved version addressing critiques.
    """
    return PromptChain(
        name="brainstorm-critique-refine",
        description="Generate ideas, critique them, then produce a refined version.",
        temperature=0.8,
        max_tokens=1024,
        steps=[
            ChainStep(
                name="brainstorm",
                system_prompt="You are a creative ideator. Think broadly and generate diverse ideas.",
                user_prompt="Generate creative ideas about:\\n\\n{{input}}",
                temperature=0.9,
                max_tokens=512,
            ),
            ChainStep(
                name="critique",
                system_prompt="You are a critical reviewer. Identify weaknesses, gaps, and risks.",
                user_prompt="Critically analyse these ideas. Point out flaws and areas for improvement:\\n\\n{{step_1}}",
                temperature=0.3,
                max_tokens=512,
            ),
            ChainStep(
                name="refine",
                system_prompt="You are a skilled synthesizer. Incorporate feedback to produce a stronger result.",
                user_prompt="Given the original ideas and the critique, produce a refined and improved version:\\n\\n"
                           "Original ideas:\\n{{step_1}}\\n\\n"
                           "Critique:\\n{{step_2}}",
                temperature=0.5,
                max_tokens=1024,
            ),
        ],
    )


def expand_and_polish() -> PromptChain:
    """Expand a rough outline into full text, then polish the prose.

    Steps:
        1. expand — Turn an outline into comprehensive text.
        2. polish — Refine the prose for clarity and flow.
    """
    return PromptChain(
        name="expand-and-polish",
        description="Expand a rough outline into full text, then polish the prose.",
        temperature=0.5,
        max_tokens=2048,
        steps=[
            ChainStep(
                name="expand",
                system_prompt="You are an expert writer. Expand the given outline into well-structured, detailed prose.",
                user_prompt="Expand this outline into full paragraphs:\\n\\n{{input}}",
                temperature=0.6,
                max_tokens=1536,
            ),
            ChainStep(
                name="polish",
                system_prompt="You are a meticulous editor. Improve clarity, flow, and word choice.",
                user_prompt="Polish this text for clarity and style:\\n\\n{{step_1}}",
                temperature=0.3,
                max_tokens=1536,
            ),
        ],
    )


def question_decomposition() -> PromptChain:
    """Break a complex question into sub-questions, answer each, then synthesise.

    Steps:
        1. decompose — Split the question into sub-questions.
        2. answer_q1 — Answer sub-question 1.
        3. answer_q2 — Answer sub-question 2.
        4. synthesise — Combine answers into a final response.
    """
    return PromptChain(
        name="question-decomposition",
        description="Break a complex question into sub-questions, answer each, then synthesise.",
        temperature=0.3,
        steps=[
            ChainStep(
                name="decompose",
                system_prompt="You are an analytical thinker. Break complex questions into simpler sub-questions.",
                user_prompt="Break this question into 2-3 simpler sub-questions that would help answer it:\\n\\n{{input}}",
                temperature=0.3,
                max_tokens=256,
            ),
            ChainStep(
                name="answer_1",
                system_prompt="You are a knowledgeable assistant. Answer the question clearly and concisely.",
                user_prompt="Answer sub-question 1 from this decomposition:\\n\\n{{step_1}}\\n\\nFocus on accuracy and brevity.",
                temperature=0.2,
                max_tokens=512,
            ),
            ChainStep(
                name="answer_2",
                system_prompt="You are a knowledgeable assistant. Answer the question clearly and concisely.",
                user_prompt="Answer sub-question 2 from this decomposition:\\n\\n{{step_1}}\\n\\nFocus on accuracy and brevity.",
                temperature=0.2,
                max_tokens=512,
            ),
            ChainStep(
                name="synthesise",
                system_prompt="You are a skilled synthesizer. Combine partial answers into a comprehensive response.",
                user_prompt="Combine these partial answers into a final, comprehensive response to the original question:\\n\\n"
                           "Original question: {{input}}\\n\\n"
                           "Sub-question answers:\\n{{step_2}}\\n\\n{{step_3}}",
                temperature=0.3,
                max_tokens=1024,
            ),
        ],
    )


def fact_check_pipeline() -> PromptChain:
    """Generate a factual claim, check evidence, then rate confidence.

    Steps:
        1. generate — Produce a factual claim based on input.
        2. evidence — List supporting or refuting evidence.
        3. rate — Assign a confidence score and reasoning.
    """
    return PromptChain(
        name="fact-check",
        description="Generate a factual claim, check evidence, then rate confidence.",
        temperature=0.2,
        steps=[
            ChainStep(
                name="generate-claim",
                system_prompt="You are a fact-checker. State a clear, specific factual claim.",
                user_prompt="Based on this topic, state one specific factual claim:\\n\\n{{input}}",
                temperature=0.3,
                max_tokens=200,
            ),
            ChainStep(
                name="evidence",
                system_prompt="You are a research assistant. List evidence for and against the claim.",
                user_prompt="For this claim, list supporting and refuting evidence:\\n\\nClaim: {{step_1}}",
                temperature=0.2,
                max_tokens=512,
            ),
            ChainStep(
                name="rate",
                system_prompt="You are a confidence assessor. Rate claim confidence based on available evidence.",
                user_prompt="Based on the claim and evidence, assign a confidence score (HIGH/MEDIUM/LOW) and explain your reasoning:\\n\\n"
                           "Claim: {{step_1}}\\n\\n"
                           "Evidence:\\n{{step_2}}",
                temperature=0.2,
                max_tokens=256,
            ),
        ],
    )


def chain_of_thought() -> PromptChain:
    """Reason step-by-step, then give a final answer.

    Steps:
        1. reason — Think through the problem step by step.
        2. answer — Provide the final answer based on the reasoning.
    """
    return PromptChain(
        name="chain-of-thought",
        description="Reason step-by-step through a problem, then provide a final answer.",
        temperature=0.3,
        steps=[
            ChainStep(
                name="reason",
                system_prompt="You are a careful reasoner. Think step by step.",
                user_prompt="Work through this problem step by step:\\n\\n{{input}}",
                temperature=0.3,
                max_tokens=1024,
            ),
            ChainStep(
                name="answer",
                system_prompt="You are a precise answerer. Give a clear, concise final answer.",
                user_prompt="Based on this reasoning, provide a final answer:\\n\\n{{step_1}}",
                temperature=0.1,
                max_tokens=512,
            ),
        ],
    )


# Registry of all templates
_BUILTIN_CHAINS = {
    "summarize-and-translate": summarize_and_translate,
    "brainstorm-critique-refine": brainstorm_critique_refine,
    "expand-and-polish": expand_and_polish,
    "question-decomposition": question_decomposition,
    "fact-check": fact_check_pipeline,
    "chain-of-thought": chain_of_thought,
}


def list_chains() -> list[str]:
    """Return names of all available chain templates."""
    return sorted(_BUILTIN_CHAINS.keys())


def load_chain(name: str) -> PromptChain:
    """Load a chain template by name.

    Raises:
        KeyError: If the name is not a registered template.
    """
    if name not in _BUILTIN_CHAINS:
        raise KeyError(f"Unknown chain template: {name!r}. "
                       f"Available: {', '.join(_BUILTIN_CHAINS)}")
    return _BUILTIN_CHAINS[name]()
