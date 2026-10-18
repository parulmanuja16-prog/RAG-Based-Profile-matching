import contextlib
import io
import json
import re
import sys
import time
from pathlib import Path

from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

# The setup module constructs the in-memory collection when imported.
with contextlib.redirect_stdout(io.StringIO()):
    from rag_setup_assignment import collection_filtered, embedding_model


class CandidateMatch(BaseModel):
    """Validate the score and evidence returned for one resume candidate."""

    candidate_name: str
    resume_path: str
    match_score: int = Field(ge=0, le=100)
    matched_skills: list[str]
    relevant_sections: list[str]
    relevant_excerpts: list[str]
    reasoning: str


class MatchResponse(BaseModel):
    """Represent the job query, ranked candidates, and retrieval metrics."""

    job_description: str
    top_matches: list[CandidateMatch]
    performance_metrics: dict[str, float | int | str | None]


def tokenize(text):
    """Split text into normalized word and technology-name tokens."""
    return re.findall(r"[a-z0-9+#.]+", text.lower())


def exact_match(phrase, text):
    """Check a phrase as a case-insensitive match bounded by word characters."""
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, re.IGNORECASE) is not None


def parse_must_haves(requirements):
    """Return the highest required experience and deduplicated skill requirements."""
    minimum_years = 0
    required_skills = []
    for requirement in (item.strip() for item in requirements.split(";")):
        if not requirement:
            continue
        years_match = re.search(r"(\d+)\s*\+?\s*(?:years?|yrs?)\b(.*)", requirement, re.IGNORECASE)
        if years_match:
            minimum_years = max(minimum_years, int(years_match.group(1)))
            skill = re.sub(r"^\s*(?:of\s+)?(?:experience\s*)?(?:with|in)?\s*", "", years_match.group(2), flags=re.IGNORECASE).strip()
            if skill:
                required_skills.append(skill)
        else:
            required_skills.append(requirement)
    return minimum_years, list(dict.fromkeys(required_skills))


def score_candidate(chunks, critical_skills, query_terms):
    """Collect exact skill matches, relevant sections, and top chunk excerpts."""
    all_text = " ".join(chunk["document"] for chunk in chunks)
    metadata = chunks[0]["metadata"]
    matched_skills = [skill for skill in critical_skills if exact_match(skill, f"{all_text} {metadata.get('skills', '')}")]
    sections = list(dict.fromkeys(
        chunk["metadata"].get("section", "Resume")
        for chunk in chunks
        if chunk["query_overlap"] or any(exact_match(skill, chunk["document"]) for skill in critical_skills)
    ))
    excerpts = [
        f"{chunk['metadata'].get('section', 'Resume')}: {chunk['document'][:300].strip()}"
        for chunk in sorted(chunks, key=lambda item: item["hybrid_score"], reverse=True)[:2]
    ]
    return matched_skills, sections, excerpts


def main():
    """Run hybrid resume retrieval, filter candidates, and print validated JSON."""
    print("Enter job description:", file=sys.stderr)
    job_description = input().strip()
    print("Enter critical skills (comma-separated, optional):", file=sys.stderr)
    critical_skills_input = input()
    critical_skills = [skill.strip() for skill in critical_skills_input.split(",") if skill.strip()]
    print("Enter must-have requirements separated by semicolons (e.g. 5+ years Python; AWS):", file=sys.stderr)
    must_haves_input = input()
    minimum_years, required_skills = parse_must_haves(must_haves_input)
    scoring_skills = list(dict.fromkeys(critical_skills + required_skills))

    retrieval_started_at = time.perf_counter()
    jd_embedding = embedding_model.encode(job_description).tolist()
    corpus = collection_filtered.get(include=["documents", "metadatas"])
    documents = corpus["documents"] or []
    metadatas = corpus["metadatas"] or []
    ids = corpus["ids"]
    if not documents:
        raise RuntimeError("The filtered resume collection is empty.")

    semantic_results = collection_filtered.query(
        query_embeddings=[jd_embedding],
        n_results=len(documents),
    )
    semantic_ranks = {
        resume_id: rank
        for rank, resume_id in enumerate(semantic_results["ids"][0], start=1)
    }
    query_terms = set(tokenize(job_description))
    tokenized_documents = [tokenize(f"{doc} {meta.get('skills', '')}") for doc, meta in zip(documents, metadatas)]
    bm25 = BM25Okapi(tokenized_documents)
    bm25_query = tokenize(f"{job_description} {' '.join(scoring_skills)}")
    bm25_scores = bm25.get_scores(bm25_query)
    bm25_order = sorted(range(len(documents)), key=lambda index: bm25_scores[index], reverse=True)
    bm25_ranks = {ids[index]: rank for rank, index in enumerate(bm25_order, start=1)}

    chunk_records = []
    for index, (resume_id, document, metadata) in enumerate(zip(ids, documents, metadatas)):
        terms_in_chunk = set(tokenize(f"{document} {metadata.get('skills', '')}"))
        query_overlap = len(query_terms & terms_in_chunk) / max(len(query_terms), 1)
        semantic_quality = 1 - (semantic_ranks.get(resume_id, len(ids)) - 1) / max(len(ids) - 1, 1)
        keyword_quality = 1 - (bm25_ranks[resume_id] - 1) / max(len(ids) - 1, 1)
        matched_critical = sum(exact_match(skill, f"{document} {metadata.get('skills', '')}") for skill in scoring_skills)
        skill_quality = matched_critical / len(scoring_skills) if scoring_skills else query_overlap
        hybrid_score = round(100 * (0.40 * semantic_quality + 0.30 * keyword_quality + 0.20 * skill_quality + 0.10 * bool(query_overlap)))
        chunk_records.append({
            "id": resume_id,
            "document": document,
            "metadata": metadata,
            "semantic_quality": semantic_quality,
            "keyword_quality": keyword_quality,
            "query_overlap": query_overlap,
            "hybrid_score": hybrid_score,
        })

    grouped = {}
    for chunk in chunk_records:
        grouped.setdefault(chunk["metadata"]["source_file"], []).append(chunk)

    candidate_matches = []
    for source_file, chunks in grouped.items():
        metadata = chunks[0]["metadata"]
        all_text = " ".join(chunk["document"] for chunk in chunks) + " " + metadata.get("skills", "")
        experience_years = int(metadata.get("experience_years", 0))
        if experience_years < minimum_years:
            continue
        if any(not exact_match(skill, all_text) for skill in required_skills):
            continue

        matched_skills, sections, excerpts = score_candidate(chunks, scoring_skills, query_terms)
        score = max(chunk["hybrid_score"] for chunk in chunks)
        reasoning_parts = [f"Hybrid semantic and BM25 relevance score: {score}/100."]
        if sections:
            reasoning_parts.append(f"Relevant resume sections: {', '.join(sections)}.")
        if matched_skills:
            reasoning_parts.append(f"Exact critical skills found: {', '.join(matched_skills)}.")
        if minimum_years:
            reasoning_parts.append(f"Meets the {minimum_years}+ years experience requirement ({experience_years} years recorded).")
        if required_skills:
            reasoning_parts.append(f"Meets required skills: {', '.join(required_skills)}.")

        candidate_matches.append(CandidateMatch(
            candidate_name=metadata.get("name", Path(source_file).stem),
            resume_path=(Path("data") / "resumes" / source_file).as_posix(),
            match_score=score,
            matched_skills=matched_skills,
            relevant_sections=sections,
            relevant_excerpts=excerpts,
            reasoning=" ".join(reasoning_parts),
        ))

    candidate_matches.sort(key=lambda candidate: candidate.match_score, reverse=True)
    top_matches = candidate_matches[:10]
    retrieval_latency_ms = round((time.perf_counter() - retrieval_started_at) * 1000, 2)
    critical_skill_relevant_count = sum(
        any(skill in candidate.matched_skills for skill in critical_skills)
        for candidate in top_matches
    )
    retrieval_accuracy = (
        round(critical_skill_relevant_count / len(top_matches), 4)
        if critical_skills and top_matches
        else None
    )
    chunk_scores = [chunk["hybrid_score"] for chunk in chunk_records]
    candidate_scores = [candidate.match_score for candidate in candidate_matches]
    response = MatchResponse(
        job_description=job_description,
        top_matches=top_matches,
        performance_metrics={
            "retrieval_latency_ms": retrieval_latency_ms,
            "retrieval_accuracy_at_k": retrieval_accuracy,
            "accuracy_basis": (
                "Fraction of returned top candidates matching at least one declared critical skill; "
                "this is a relevance proxy, not ground-truth benchmark accuracy."
                if critical_skills
                else "Unavailable: no critical skills were declared."
            ),
            "relevant_candidates_at_k": critical_skill_relevant_count if critical_skills else None,
            "evaluated_candidates_at_k": len(top_matches),
            "total_chunks": len(chunk_records),
            "total_candidates": len(grouped),
            "average_chunk_score": round(sum(chunk_scores) / len(chunk_scores), 2) if chunk_scores else None,
            "average_candidate_score": round(sum(candidate_scores) / len(candidate_scores), 2) if candidate_scores else None,
            "minimum_chunk_score": min(chunk_scores) if chunk_scores else None,
            "maximum_chunk_score": max(chunk_scores) if chunk_scores else None,
            "minimum_candidate_score": min(candidate_scores) if candidate_scores else None,
            "maximum_candidate_score": max(candidate_scores) if candidate_scores else None,
        },
    )
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
