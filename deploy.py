"""
deploy.py — Momento Vertex AI Deployment
=========================================
Deploys three agents to Vertex AI Agent Engine:

  1. DecomposeAgentApp     — wraps decomposing_agent.run_decomposer()
  2. CompatibilityAgentApp — wraps the full decompose → score → aggregate pipeline
  3. ProfileAgentApp       — wraps run_compatibility_for_all() (batch, all passages)

The compatibility pipeline is:
  decompose_moment (Gemini call per reader)
  → score_compatibility (Gemini call per pair)
  → aggregate() (pure Python)
  → R/C/D percentages + confidence

Run locally: python deploy.py
Run in CI:   called by the deploy job in cicd.yml
"""

import os
import vertexai # type: ignore
from vertexai import agent_engines # type: ignore

from model_interface import (
    decompose_moment,
    run_compatibility_pipeline,
    run_batch_compatibility,
)

# ── Init ──────────────────────────────────────────────────────────────────────
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "moment-486719")
REGION     = os.environ.get("GCP_REGION",     "us-central1")
BUCKET     = os.environ.get("GCS_BUCKET",     "gs://moment-agent-data")

vertexai.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET)

REQUIREMENTS = [
    "google-cloud-aiplatform[agent_engines]",
    "google-genai",
    "google-cloud-storage",
    "python-dotenv",
    "pydantic==2.12.5",
    "cloudpickle==3.1.2",
]

LOCAL_PACKAGES = [
    "tools.py",
    "decomposing_agent.py",
    "compatibility_agent.py",
    "aggregator.py",
    "model_interface.py",
]

# ── Agent 1: Decompose ────────────────────────────────────────────────────────

class DecomposeAgentApp:
    """
    Wraps the Decomposition Agent.
    Called when a user writes or updates a moment — decomposes it into
    weighted sub-claims with emotional modes.

    Input:  passage_id, user_id, moment_text, word_count, book_id
    Output: decomposition dict (subclaims + weights + emotional modes)
    """
    def query(
        self,
        user_id: str,
        passage_id: str,
        book_id: str,
        moment_text: str,
        word_count: int = 0,
    ) -> dict:
        return decompose_moment(
            passage_id=passage_id,
            user_id=user_id,
            moment_text=moment_text,
            word_count=word_count,
            book_id=book_id,
        )


# ── Agent 2: Compatibility (per passage pair) ─────────────────────────────────

class CompatibilityAgentApp:
    """
    Wraps the full compatibility pipeline for one user pair on one passage.
    Pipeline: decompose A → decompose B → score → aggregate → R/C/D + confidence.

    Decompositions are cached automatically in data/processed/decompositions.json.
    Scoring runs are cached in data/processed/scoring_runs.json.

    Input:  user_a_id, user_b_id, book_id, passage_id, moments_map
    Output: compatibility result dict
    """
    def query(
        self,
        user_a_id: str,
        user_b_id: str,
        book_id: str,
        passage_id: str,
        moments_map: dict,
    ) -> dict:
        moment_a = moments_map.get(user_a_id, {})
        moment_b = moments_map.get(user_b_id, {})
        return run_compatibility_pipeline(
            user_a=user_a_id,
            user_b=user_b_id,
            book=book_id,
            passage_id=passage_id,
            moment_a=moment_a,
            moment_b=moment_b,
        )


# ── Agent 3: Profile (batch — all passages for a user pair) ──────────────────

class ProfileAgentApp:
    """
    Wraps the batch compatibility runner.
    Scores one anchor user against all other users on a given passage.
    Results are sorted by confidence and cached automatically.

    Input:  user_a_id, book_id, passage_id, moments_map
    Output: list of compatibility result dicts (sorted by confidence desc)
    """
    def query(
        self,
        user_a_id: str,
        book_id: str,
        passage_id: str,
        moments_map: dict,
    ) -> list:
        return run_batch_compatibility(
            user_a_id=user_a_id,
            book_id=book_id,
            passage_id=passage_id,
            moments_map=moments_map,
        )


# ── Deploy ────────────────────────────────────────────────────────────────────

def deploy_all():
    print(f"Deploying to project={PROJECT_ID} region={REGION} bucket={BUCKET}")

    decompose_remote = agent_engines.create(
        DecomposeAgentApp(),
        requirements=REQUIREMENTS,
        extra_packages=LOCAL_PACKAGES,
        display_name="momento-decompose-agent",
    )
    print(f"Decompose Agent deployed:      {decompose_remote.resource_name}")

    compat_remote = agent_engines.create(
        CompatibilityAgentApp(),
        requirements=REQUIREMENTS,
        extra_packages=LOCAL_PACKAGES,
        display_name="momento-compatibility-agent",
    )
    print(f"Compatibility Agent deployed:  {compat_remote.resource_name}")

    profile_remote = agent_engines.create(
        ProfileAgentApp(),
        requirements=REQUIREMENTS,
        extra_packages=LOCAL_PACKAGES,
        display_name="momento-profile-agent",
    )
    print(f"Profile Agent deployed:        {profile_remote.resource_name}")

    return {
        "decompose_agent":     decompose_remote.resource_name,
        "compatibility_agent": compat_remote.resource_name,
        "profile_agent":       profile_remote.resource_name,
    }


if __name__ == "__main__":
    deployed = deploy_all()
    print("\nAll agents deployed:")
    for name, resource in deployed.items():
        print(f"  {name}: {resource}")