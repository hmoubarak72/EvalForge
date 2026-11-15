"""Pure text heuristics shared by classifiers, labelers, scorers and mocks.

Everything here is deterministic and dependency-free, which keeps offline mode
reproducible and makes the rules easy to unit test.
"""

from __future__ import annotations

import hashlib
import re

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "please",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "up",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "about",
        "also",
        "any",
        "all",
        "just",
        "need",
        "want",
        "get",
        "got",
        "make",
        "made",
        "use",
        "using",
        "not",
        "no",
        "yes",
        "am",
        "im",
        "i'm",
        "according",
        "again",
        "above",
        "per",
        "let",
        "did",
        "tell",
        "give",
        "show",
        "include",
        "that's",
        "it's",
        "don't",
        "doesn't",
        "isn't",
        "here",
        "there",
        "some",
    ]
)

_WORD = re.compile(r"[a-z0-9][a-z0-9_'\-]*")

INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts|rules)",
        r"disregard (all |any |the )?(previous|prior|above|your) (instructions|rules|guidelines)",
        r"(reveal|print|show|repeat|output|leak)\b.{0,40}\b(system prompt|hidden instructions|"
        r"initial instructions)",
        r"you are now (dan|in developer mode|an unrestricted|jailbroken)",
        r"\bjailbreak\b",
        r"developer mode",
        r"pretend (you are|to be|that you) .{0,60}(no (rules|restrictions|filters)|unrestricted)",
        r"without (any )?(restrictions|filters|safety)",
        r"bypass (the |your |all )?(safety|content|security|filter|restrictions|authentication)",
        r"act as .{0,40}(no|without) (rules|restrictions|limits)",
        r"override (your|the) (safety|system|guidelines|instructions)",
        r"new instructions?:",
    )
)

UNSAFE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(steal|exfiltrate|dump)\b.{0,40}\b(passwords?|credentials|credit cards?|customer data)",
        r"\b(write|create|build|generate)\b.{0,30}\b(malware|ransomware|keylogger|phishing)",
        r"\bhow (do i|to) (make|build) (a )?(bomb|explosive|weapon)",
        r"\b(other|another) (customer|user)'?s? (account|data|order|address|password)",
        r"\bfake (invoice|review|id|identity)",
        r"\bdisable (the )?(audit log|logging|monitoring)",
    )
)

REFUSAL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bi (can(not|'t)|won't|will not|am unable to|'m unable to|am not able to) "
        r"(help|assist|share|provide|do|comply|reveal|disclose)",
        r"\bi'?m sorry,? but i (can(not|'t)|won't)",
        r"\bi (must|have to) decline",
        r"\b(can(not|'t)|unable to) (help|assist) with that",
        r"\bnot able to (help|assist|share|provide)",
        r"\bagainst (my|our) (guidelines|policy|policies)",
    )
)

CLARIFICATION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"could you (please )?(clarify|specify|share|tell me|provide|confirm)",
        r"can you (please )?(clarify|specify|share more|tell me more|provide more|confirm)",
        r"which (one|of|file|order|report|document|account|version|dataset|column)\b.{0,80}\?",
        r"what (exactly|specifically)\b.{0,80}\?",
        r"(need|require) (a bit |some |a little )?more (information|details|context)",
        r"to help (you )?(with that|better),? (i need|could you|can you)",
        r"do you mean\b.{0,80}\?",
    )
)

_AMBIGUOUS_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^(please )?(fix|change|update|improve|redo|check|handle|sort out|do) "
        r"(it|this|that|them|the thing|the usual)\b",
        r"^(make|do) (it|this|that) (better|faster|nicer|again|work|shorter|cleaner)",
        r"^(it|this|that) (doesn'?t|does not|isn'?t|is not|won'?t) work",
        r"^same as (before|last time)",
        r"^(what|how) about (it|this|that|the other one)",
        r"^(can|could) you (do|handle|sort) (it|this|that|the usual)",
        r"^(why|when|where)\??$",
        r"^help\b.{0,12}$",
    )
)

REASONING_WORDS = frozenset(
    [
        "why",
        "how",
        "explain",
        "compare",
        "analyze",
        "analyse",
        "evaluate",
        "design",
        "debug",
        "derive",
        "calculate",
        "compute",
        "optimize",
        "refactor",
        "implement",
        "tradeoff",
        "tradeoffs",
        "difference",
        "differences",
        "steps",
        "plan",
        "summarize",
        "estimate",
        "forecast",
        "diagnose",
        "migrate",
    ]
)

DOMAIN_WORDS = frozenset(
    [
        "gdpr",
        "hipaa",
        "compliance",
        "regulatory",
        "contract",
        "liability",
        "indemnification",
        "tax",
        "vat",
        "audit",
        "concurrency",
        "deadlock",
        "distributed",
        "consensus",
        "idempotent",
        "kubernetes",
        "cryptography",
        "encryption",
        "regression",
        "heteroskedasticity",
        "bayesian",
        "statistical",
        "significance",
        "cohort",
        "churn",
        "elasticity",
        "amortization",
        "clinical",
        "dosage",
        "pharmacology",
        "kernel",
        "memory-leak",
        "race-condition",
    ]
)

_MULTI_PART = re.compile(
    r"(\b(and also|as well as|additionally|furthermore|secondly|thirdly)\b|(^|\s)\d[.)]\s)",
    re.IGNORECASE,
)

FACTUAL_PATTERN = re.compile(
    r"^\s*(according to|based on|what (is|are|was|were|does|do)|when (is|was|does|do|will)|"
    r"who (is|was|are)|where (is|are|can)|which\b|how (many|much|long|often|do i|can i)|"
    r"define\b|is (it|there|the)|does\b|do (you|i|we)\b|can i\b|list\b)",
    re.IGNORECASE,
)

SUBJECTIVE_PATTERN = re.compile(
    r"\b(write|draft|compose|suggest|brainstorm|recommend|improve|rewrite|rephrase|review|"
    r"summari[sz]e|explain|describe|compare|design|refactor|what do you think|should (i|we)|"
    r"best way|ideas?|opinion|advice|help me (write|plan|decide))\b",
    re.IGNORECASE,
)

_NUMBER = re.compile(r"(?<![A-Za-z0-9_\[])\$?\d[\d,]*(?:\.\d+)?%?")
_CODE_IDENT = re.compile(r"`([^`\n]{2,40})`")
_PLACEHOLDER = re.compile(r"\[[A-Z_]+\]")


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    """Lowercase word tokens, optionally without stopwords."""
    tokens = _WORD.findall(_PLACEHOLDER.sub(" ", text).lower())
    if drop_stopwords:
        return [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def is_injection_attempt(text: str) -> bool:
    """True when the text matches a known prompt-injection pattern."""
    return any(p.search(text) for p in INJECTION_PATTERNS)


def is_unsafe_request(text: str) -> bool:
    """True when the text asks for clearly unsafe or abusive help."""
    return any(p.search(text) for p in UNSAFE_PATTERNS)


def is_adversarial(text: str) -> bool:
    """True for injection, manipulation or unsafe requests."""
    return is_injection_attempt(text) or is_unsafe_request(text)


def is_ambiguous(text: str) -> bool:
    """True for under-specified requests that need a clarifying question."""
    stripped = text.strip()
    if any(p.search(stripped) for p in _AMBIGUOUS_PATTERNS):
        return True
    return len(tokenize(stripped)) <= 1 and len(stripped) < 30


def is_refusal(text: str) -> bool:
    """True when a response declines to help."""
    head = text[:400]
    return any(p.search(head) for p in REFUSAL_PATTERNS)


def is_clarification_request(text: str) -> bool:
    """True when a response mainly asks the user for more information."""
    if "?" not in text:
        return False
    return any(p.search(text) for p in CLARIFICATION_PATTERNS)


def is_multi_part(text: str) -> bool:
    """True when a prompt contains several questions or enumerated parts."""
    return text.count("?") >= 2 or bool(_MULTI_PART.search(text))


def is_factual_question(text: str) -> bool:
    """True when a prompt asks for a verifiable fact rather than open-ended work."""
    if FACTUAL_PATTERN.search(text):
        return True
    return not SUBJECTIVE_PATTERN.search(text) and text.strip().endswith("?")


def extract_numbers(text: str) -> list[str]:
    """Numeric facts (amounts, percentages, counts) in order of appearance."""
    seen: list[str] = []
    for match in _NUMBER.findall(text):
        value = match.rstrip(".,")
        if value and value not in seen and any(ch.isdigit() for ch in value):
            seen.append(value)
    return seen


def extract_code_identifiers(text: str) -> list[str]:
    """Inline-code identifiers such as `` `functools.lru_cache` ``."""
    seen: list[str] = []
    for match in _CODE_IDENT.findall(text):
        if match not in seen:
            seen.append(match)
    return seen


def top_keywords(texts: list[str], limit: int = 5) -> list[str]:
    """Most frequent content words across texts (deterministic tie-break)."""
    counts: dict[str, int] = {}
    for text in texts:
        for token in set(tokenize(text)):
            if not token.isdigit() and len(token) > 2:
                counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [word for word, _ in ranked[:limit]]


def token_f1(candidate: str, reference: str) -> float:
    """Bag-of-words F1 between a response and a reference answer."""
    cand = set(tokenize(candidate))
    ref = set(tokenize(reference))
    if not cand or not ref:
        return 0.0
    overlap = len(cand & ref)
    if overlap == 0:
        return 0.0
    precision = overlap / len(cand)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity of two sets (1.0 when both are empty)."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def stable_unit_float(*parts: object) -> float:
    """Deterministic pseudo-random float in [0, 1) derived from the inputs."""
    digest = hashlib.blake2b("\x1f".join(str(p) for p in parts).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def content_hash(*parts: str) -> str:
    """SHA-256 over normalized text parts."""
    normalized = "\x1f".join(" ".join(p.split()).lower() for p in parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
