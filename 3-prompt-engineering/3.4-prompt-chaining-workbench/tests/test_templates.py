"""Tests for pre-built chain templates."""

from prompt_chaining.templates import (
    summarize_and_translate,
    brainstorm_critique_refine,
    expand_and_polish,
    question_decomposition,
    fact_check_pipeline,
    chain_of_thought,
    list_chains,
    load_chain,
)


class TestChainTemplates:
    def test_summarize_and_translate(self):
        chain = summarize_and_translate()
        assert chain.name == "summarize-and-translate"
        assert chain.step_count == 2
        assert chain[0].name == "summarise"
        assert chain[1].name == "translate"
        assert "{{step_1}}" in chain[1].user_prompt

    def test_brainstorm_critique_refine(self):
        chain = brainstorm_critique_refine()
        assert chain.name == "brainstorm-critique-refine"
        assert chain.step_count == 3
        assert chain[0].name == "brainstorm"
        assert chain[1].name == "critique"
        assert chain[2].name == "refine"
        assert chain[0].temperature == 0.9
        assert chain[1].temperature == 0.3

    def test_expand_and_polish(self):
        chain = expand_and_polish()
        assert chain.name == "expand-and-polish"
        assert chain.step_count == 2
        assert chain[0].name == "expand"
        assert chain[1].name == "polish"

    def test_question_decomposition(self):
        chain = question_decomposition()
        assert chain.name == "question-decomposition"
        assert chain.step_count == 4
        assert chain[0].name == "decompose"
        assert chain[3].name == "synthesise"
        assert "{{input}}" in chain[3].user_prompt

    def test_fact_check_pipeline(self):
        chain = fact_check_pipeline()
        assert chain.name == "fact-check"
        assert chain.step_count == 3
        assert chain[0].name == "generate-claim"
        assert chain[1].name == "evidence"
        assert chain[2].name == "rate"

    def test_chain_of_thought(self):
        chain = chain_of_thought()
        assert chain.name == "chain-of-thought"
        assert chain.step_count == 2
        assert chain[0].name == "reason"
        assert chain[1].name == "answer"
        assert chain[1].temperature == 0.1

    def test_all_templates_have_unique_names(self):
        chains = [summarize_and_translate(), brainstorm_critique_refine(),
                  expand_and_polish(), question_decomposition(),
                  fact_check_pipeline(), chain_of_thought()]
        names = [c.name for c in chains]
        assert len(names) == len(set(names))

    def test_all_templates_have_description(self):
        chains = [summarize_and_translate(), brainstorm_critique_refine(),
                  expand_and_polish(), question_decomposition(),
                  fact_check_pipeline(), chain_of_thought()]
        for c in chains:
            assert c.description, f"Chain {c.name} has no description"

    def test_all_step_user_prompts_are_not_empty(self):
        chains = [summarize_and_translate(), brainstorm_critique_refine(),
                  expand_and_polish(), question_decomposition(),
                  fact_check_pipeline(), chain_of_thought()]
        for c in chains:
            for step in c.steps:
                assert step.user_prompt, f"Step {step.name} in {c.name} has empty user_prompt"


class TestListChains:
    def test_list_chains(self):
        names = list_chains()
        assert len(names) == 6
        assert "summarize-and-translate" in names
        assert "chain-of-thought" in names

    def test_load_chain(self):
        chain = load_chain("chain-of-thought")
        assert chain.name == "chain-of-thought"

    def test_load_unknown(self):
        import pytest
        with pytest.raises(KeyError, match="Unknown chain"):
            load_chain("does-not-exist")

    def test_list_is_sorted(self):
        names = list_chains()
        assert names == sorted(names)
