"""
generate_adjust_prompt_pc59.py
--------------------------------
Generates adjust_prompt_pc59.json (short SAM3 prompt lists) from contexts_class_pc59.json,
using an LLM. Adapted from generate_adjust_prompt_voc_v2.py / generate_adjust_prompt_city_v2.py
(the validated version used for City/VOC) -- ONLY the default --input/--output filenames
changed for PC59; every rule and the anchor_prompts/generation_notes handling are otherwise
byte-identical to the version already used and validated for City/VOC.

Input : contexts_class_pc59.json  {class: {definition, positive, negative, confusable_classes,
                                            anchor_prompts?, generation_notes?}}
Output: adjust_prompt_pc59.json   {class: [prompt1, prompt2, ...]}
        adjust_prompt_pc59_meta.json  (raw response + config, for paper reproducibility)

API key:
    export GEMINI_API_KEY="..."
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types, errors
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
# Schema forcing the LLM to return correctly-shaped JSON
# --------------------------------------------------------------------------- #
class ClassPromptItem(BaseModel):
    class_name: str = Field(description="Class name, must exactly match the input key.")
    prompts: list[str] = Field(
        description=(
            "Short prompts for open-vocabulary segmentation, typically 2-4 total "
            "(the class's generation_notes may explicitly ask for even fewer, "
            "precision-first). Strongly prefer a single noun; use a 2-word phrase "
            "only when one word loses essential meaning. No full sentences, no "
            "articles."
        )
    )


class AdjustPromptResponse(BaseModel):
    items: list[ClassPromptItem]


# --------------------------------------------------------------------------- #
# Prompt construction for the LLM
# --------------------------------------------------------------------------- #
SYSTEM_INSTRUCTION = """\
You are a prompt-engineering expert for open-vocabulary segmentation models such as \
SAM3/CLIP (their text encoders match short phrases much better than long, \
grammatically complex sentences).

For each class provided (definition, positive, negative, confusable_classes, and \
optionally anchor_prompts / generation_notes), generate prompts to be used for \
querying the segmentation model, strictly following these rules, IN ORDER OF PRIORITY:

1. If the class context includes "anchor_prompts": these terms MUST be copied \
   VERBATIM into the output prompt list. They are strong, well-established terms \
   that must never be dropped, reworded, or replaced by a narrower synonym -- even if \
   a "positive" entry seems to cover similar ground. Silently dropping an anchor term \
   in favor of a "more specific" synonym is the single most common mistake to avoid.
2. If "generation_notes" is present for a class, treat it as a HARD CONSTRAINT that \
   overrides every other rule below where they conflict (including the target count \
   in rule 3). Read it carefully -- it documents a specific failure mode already \
   observed for that class.
3. Unless generation_notes says otherwise, produce 2-4 total prompts per class \
   (anchor_prompts count toward this total, plus enough additional discriminative \
   variants to reach it, but never exceed 4). Fewer, stronger prompts beat padding \
   the list with weak or redundant synonyms.
4. STRONGLY prefer a single noun (e.g. "grass", "lawn", "sand"). Only use a 2-word \
   phrase when a single word would lose essential meaning (e.g. "sign pole" instead \
   of just "pole" if "pole" alone would be too easily confused with objects in \
   confusable_classes).
5. Simplify/shorten entries from the given "positive" list -- do NOT invent concepts \
   outside the scope of "definition". Drop concepts that are too abstract and lack a \
   clear visual form (e.g. "horizontal vegetation" is an abstract description, not a \
   good prompt).
6. NEVER generate a prompt that duplicates or is synonymous with any entry in that \
   same class's own "negative" list.
7. Avoid prompts that could be easily confused with the listed "confusable_classes" -- \
   prefer words with high discriminative power against those classes.
8. Avoid generic, broad descriptors that pair a common material/size/room adjective \
   with a generic noun (e.g. "wooden table", "large table", "upholstered furniture") \
   unless the class's own defining noun is present in the phrase -- these tend to \
   match many unrelated objects and cause over-segmentation (predicted area much \
   larger than ground truth).
9. Output must be standalone nouns/noun phrases, not full sentences, no articles \
   (a/an/the), no markdown, no extra explanation.

Respond for ALL classes provided in a single pass, keeping "class_name" identical to \
the original input key.
"""


def build_user_content(class_contexts: dict) -> str:
    """Serialize contexts_class_pc59.json into the request content. Keep the original
    structure as-is (no rewriting) so the LLM sees exactly the definition/positive/
    negative/confusable/anchor_prompts/generation_notes fields we hand-curated -- this
    is the constraint mechanism that stands in for self-consistency sampling.
    """
    return (
        "List of classes to generate prompts for (JSON):\n\n"
        f"{json.dumps(class_contexts, ensure_ascii=False, indent=2)}"
    )


# --------------------------------------------------------------------------- #
# API call with retry for transient errors (429 rate-limit / 5xx)
# --------------------------------------------------------------------------- #
def call_gemini_with_retry(
    client: genai.Client,
    model: str,
    contents: str,
    config: types.GenerateContentConfig,
    max_retries: int = 5,
    base_delay: float = 10.0,
) -> types.GenerateContentResponse:
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except errors.APIError as e:
            last_err = e
            if e.code == 429 or 500 <= e.code < 600:
                delay = base_delay * (2**attempt)
                print(
                    f"[warn] API error {e.code} (attempt {attempt + 1}/{max_retries}), "
                    f"backing off {delay:.0f}s before retry...",
                    file=sys.stderr,
                )
                time.sleep(delay)
                continue
            raise
    raise RuntimeError(f"Exhausted {max_retries} retries, last error: {last_err}") from last_err


# --------------------------------------------------------------------------- #
# Offline validation: catch concepts the LLM self-contradicts against its own
# negative list, and enforce that anchor_prompts always survive
# --------------------------------------------------------------------------- #
def _normalize(text: str) -> str:
    return text.strip().lower()


def validate_and_clean(
    class_contexts: dict, generated: dict[str, list[str]]
) -> tuple[dict[str, list[str]], list[str]]:
    """
    - Hard-drop a prompt if it exactly matches (after normalization) an entry in
      that class's own negative list -- this is an unambiguous LLM bug, not a
      judgment call, so it's dropped automatically.
    - Only WARN (do not auto-remove) if a prompt matches a positive concept of a
      confusable class, since that overlap can sometimes be legitimate depending
      on context -- left for manual review.
    - Enforce "anchor_prompts": if the LLM dropped one, auto-prepend it rather than
      just warning. This is the exact failure mode that caused sofa/chair/bicycle to
      regress in practice (the bare class term silently replaced by a narrower
      synonym), so it is corrected automatically instead of relying on a manual pass.
    """
    cleaned: dict[str, list[str]] = {}
    warnings: list[str] = []

    for class_name, prompts in generated.items():
        ctx = class_contexts.get(class_name, {})
        own_negative = {_normalize(n) for n in ctx.get("negative", [])}
        confusable = ctx.get("confusable_classes", [])
        confusable_positive = {
            _normalize(p)
            for other in confusable
            for p in class_contexts.get(other, {}).get("positive", [])
        }
        anchors = ctx.get("anchor_prompts", [])

        kept: list[str] = []
        seen_norm: set[str] = set()
        for p in prompts:
            p_norm = _normalize(p)
            if p_norm in own_negative:
                warnings.append(
                    f"[DROP] '{class_name}': prompt '{p}' matches its own negative list -> removed."
                )
                continue
            if p_norm in confusable_positive:
                warnings.append(
                    f"[REVIEW] '{class_name}': prompt '{p}' matches a positive concept of a "
                    f"confusable class -> kept, but needs manual review."
                )
            if p_norm not in seen_norm:
                kept.append(p)
                seen_norm.add(p_norm)

        for anchor in anchors:
            anchor_norm = _normalize(anchor)
            if anchor_norm not in seen_norm:
                warnings.append(
                    f"[FIX] '{class_name}': anchor prompt '{anchor}' was missing from the "
                    f"LLM output -> auto-added."
                )
                kept.insert(0, anchor)
                seen_norm.add(anchor_norm)

        if not kept:
            warnings.append(
                f"[EMPTY] '{class_name}': all prompts were removed, needs manual regeneration."
            )
        cleaned[class_name] = kept

    return cleaned, warnings


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def generate_adjust_prompts(
    input_path: str,
    output_path: str,
    model: str = "gemini-3.1-flash-lite",
    temperature: float = 0.3,
) -> dict[str, list[str]]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Missing GEMINI_API_KEY. Set it via `export GEMINI_API_KEY=..."
        )

    class_contexts = json.loads(Path(input_path).read_text(encoding="utf-8"))

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=AdjustPromptResponse,
    )

    response = call_gemini_with_retry(
        client=client,
        model=model,
        contents=build_user_content(class_contexts),
        config=config,
    )

    parsed: AdjustPromptResponse = response.parsed
    if parsed is None:
        raise ValueError(f"Failed to parse structured output. Raw text: {response.text!r}")

    generated = {item.class_name: item.prompts for item in parsed.items}

    # Check for missing classes (LLM skipped one) -- fail fast instead of silently
    # missing prompts later during training.
    missing = set(class_contexts.keys()) - set(generated.keys())
    if missing:
        raise ValueError(f"LLM did not return prompts for classes: {missing}. Re-run.")

    cleaned, warnings = validate_and_clean(class_contexts, generated)
    for w in warnings:
        print(w, file=sys.stderr)

    Path(output_path).write_text(
        json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Save raw response + config for the paper's reproducibility appendix.
    meta_path = str(Path(output_path).with_name(Path(output_path).stem + "_meta.json"))
    meta = {
        "model": model,
        "temperature": temperature,
        "input_file": input_path,
        "raw_response_text": response.text,
        "validation_warnings": warnings,
    }
    Path(meta_path).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return cleaned


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate adjust_prompt_pc59.json from contexts_class_pc59.json via Gemini."
    )
    parser.add_argument("--input", default="contexts_class_pc59.json")
    parser.add_argument("--output", default="adjust_prompt_pc59.json")
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()

    result = generate_adjust_prompts(args.input, args.output, args.model, args.temperature)
    print(json.dumps(result, ensure_ascii=False, indent=2))