"""
Headline relevance scoring -- upgrades the news check from "does any
headline exist for this ticker nearby" to "does this headline actually
read like the kind of event that would explain a sudden price move".

This is the literal wording of the problem statement's hardest clause ("a
price jump with no news behind it"): presence alone treats a routine or
unrelated headline the same as a genuine market-moving one. Deliberately
dependency-light -- TF-IDF + cosine similarity against a small reference
set of market-moving event archetypes, plus a keyword/entity boost -- not
a heavy NLP/sentiment stack, for the same free-tier-hosting reason the
project uses an MLP instead of a transformer elsewhere.
"""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer

# Short reference set of market-moving event archetypes. A headline that
# reads like one of these is plausible evidence for an unusual price move;
# a headline that doesn't (e.g. a routine "analyst maintains rating" wire
# story) shouldn't get full credit just because it mentions the ticker.
MARKET_MOVING_EVENTS = [
    "earnings report beats or misses analyst expectations",
    "merger or acquisition announcement",
    "lawsuit or regulatory investigation",
    "trading halt or circuit breaker triggered",
    "revised guidance or profit warning issued",
    "credit rating upgrade or downgrade",
    "product recall or safety issue disclosed",
    "executive resignation or leadership change",
    "bankruptcy or insolvency filing",
    "fda approval or clinical trial result",
    "data breach or security incident",
    "stock split, buyback, or dividend change announced",
]

# A fast keyword/entity check as a floor under the similarity score --
# catches short, terse headlines that TF-IDF cosine similarity (built for
# longer documents) can under-score.
_KEYWORDS = {
    "earnings", "merger", "acquisition", "lawsuit", "investigation", "halt",
    "guidance", "downgrade", "upgrade", "recall", "resign", "resignation",
    "bankruptcy", "fda", "breach", "hack", "hacked", "dividend", "split",
    "buyback", "fraud", "settlement", "layoffs", "restructuring", "warning",
    "sec", "probe", "indictment", "default", "delisted", "halted",
}

_vectorizer: TfidfVectorizer | None = None
_reference_matrix = None


def _get_vectorizer():
    global _vectorizer, _reference_matrix
    if _vectorizer is None:
        _vectorizer = TfidfVectorizer(stop_words="english")
        _reference_matrix = _vectorizer.fit_transform(MARKET_MOVING_EVENTS)
    return _vectorizer, _reference_matrix


def score_headline_relevance(headline: str | None, ticker: str | None = None) -> float:
    """0.0-1.0 relevance score for whether `headline` plausibly explains a
    price move in `ticker` -- not just whether a headline exists at all.

    Blend: TF-IDF cosine similarity to the market-moving-event reference
    set (70% weight) + a keyword/entity hit boost (up to +0.35) + a small
    boost if the ticker itself is named (+0.10, since a bare mention isn't
    much evidence on its own). Clipped to [0, 1].
    """
    if not headline or not str(headline).strip():
        return 0.0
    text = str(headline).lower()

    vectorizer, reference_matrix = _get_vectorizer()
    try:
        vec = vectorizer.transform([text])
        sims = (vec @ reference_matrix.T).toarray().ravel()
        similarity = float(sims.max()) if len(sims) else 0.0
    except Exception:
        similarity = 0.0

    keyword_hit = any(re.search(rf"\b{re.escape(kw)}\b", text) for kw in _KEYWORDS)
    keyword_boost = 0.35 if keyword_hit else 0.0

    ticker_boost = 0.0
    if ticker and re.search(rf"\b{re.escape(str(ticker).lower())}\b", text):
        ticker_boost = 0.10

    score = similarity * 0.70 + keyword_boost + ticker_boost
    return round(min(1.0, max(0.0, score)), 3)
