# Repository Instructions

## `ml-zoomcamp/learning-qa.md`

When the user asks to summarize or land chat content into `learning-qa.md`, write concise Chinese Q&A notes that preserve high-value learning points.

Prefer recording:

- Mechanism-level understanding, such as how an algorithm works, why a formula is used, or how a training/evaluation process is structured.
- Questions that expose confusion or a likely blind spot, especially where the answer changes how to reason about the model.
- Practical pitfalls that caused or could cause wrong results, such as data leakage, wrong scaling scope, incompatible library behavior, environment/kernel issues, or version-specific API errors.
- Small realistic examples that make an abstract idea concrete, especially when they include the minimal math or step-by-step process needed to understand it.

Avoid recording:

- Simple field definitions, one-line API meanings, or generic factual answers unless they are non-obvious, caused confusion, or connect to a real modeling pitfall.
- Long conversational explanations, duplicate points, or low-value glossary entries.

Style:

- Keep each chapter as a dedicated section, e.g. `## 06-trees 核心问答`.
- Keep Q&A entries short and focused, but include enough math/example detail for mechanism questions.
- Prefer precise distinctions, such as hard classification output vs internal probability distribution, validation vs test use, or feature selection vs threshold selection.
- Do not be verbose.
