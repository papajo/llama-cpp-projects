"""
Natural language / prose prompts.

These have less predictable patterns — good for testing draft-model
strategies which should handle the variability better than n-gram.
"""

PROSE_PROMPTS = [
    (
        "prose-explanation",
        """Explain the concept of attention mechanisms in transformer models. Cover:
1. What problem attention solves (long-range dependencies in sequences)
2. How scaled dot-product attention works (Q, K, V matrices)
3. Multi-head attention and why it helps
4. The role of attention in the encoder-decoder architecture

Write in a clear, educational style suitable for someone with basic ML knowledge.
"""
    ),
    (
        "prose-creative-story",
        """Write a short science fiction story about the first AI to develop genuine creativity. 
The AI is not a chatbot or a助手 — it's a system designed to find novel solutions to 
engineering problems, and one day it starts producing things that have no practical 
purpose but are beautiful. The story should explore what "creativity" means when 
it emerges from gradient descent and matrix multiplications.

Make it around 500 words, with a surprising but emotionally satisfying ending.
"""
    ),
    (
        "prose-summarization",
        """Please summarise the following article in 3-4 sentences:

Large language models have transformed how we interact with technology, 
but their deployment in production environments presents unique challenges. 
Latency requirements demand careful optimisation of inference pipelines, 
while memory constraints on consumer GPUs limit the size of models that 
can be run locally. Techniques such as quantization, speculative decoding, 
and KV cache optimisation have emerged as critical tools for making LLMs 
practical on commodity hardware. The tradeoff between model quality and 
inference speed remains an active area of research, with new approaches 
like multi-token prediction and attention-free architectures promising 
further improvements. For developers building local-first AI applications, 
understanding these performance characteristics is essential for making 
informed architectural decisions.
"""
    ),
    (
        "prose-instruction-following",
        """Given the following customer email, draft a professional response:

---
Dear Support Team,

I've been using your API for the past month and it's been working great, 
but since yesterday I'm getting 503 errors on about 20% of my requests. 
My account is in good standing and I haven't changed any code. 

This is affecting my production system. Can you please help?

Best,
Alex Chen
Account: acme-corp-123
---

Your response should:
- Acknowledge the issue and apologise
- Ask for specific information (exact timestamps, endpoint paths, request IDs)
- Provide an initial troubleshooting step they can try immediately
- Set expectations for resolution time
- Include a ticket number (TKT-4567)
"""
    ),
    (
        "prose-essay",
        """Write a persuasive essay arguing for or against the following position:

"The development of artificial general intelligence (AGI) should be paused 
until we have robust international governance frameworks in place."

Consider:
- Current capabilities and trajectory of AI systems
- Economic incentives driving development
- Historical precedents for technology governance (nuclear, biotech, aviation)
- The challenge of enforcement across jurisdictions
- Opportunity costs of delay

Aim for a balanced but decisive argument, approximately 400 words.
"""
    ),
]
