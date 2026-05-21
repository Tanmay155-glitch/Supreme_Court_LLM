"""
ULI — Phase 2D: Pinecone namespace configuration.
Index spec, namespace constants, NAMESPACE_BRANCH_MAP, and async upsert helper.
"""
from __future__ import annotations

import asyncio
import os
from typing import Dict, List

from pinecone import Pinecone, ServerlessSpec

from uli.models import Section, SubBranch

# ─────────────────────────────────────────────────────────────────────────────
# INDEX SPECIFICATION
# ─────────────────────────────────────────────────────────────────────────────

INDEX_NAME   = "uli-legal-index"
DIMENSION    = 1536                   # OpenAI text-embedding-3-small / ada-002
METRIC       = "cosine"
CLOUD        = "aws"
REGION       = "ap-south-1"           # Mumbai — lowest latency for IN judiciary


# ─────────────────────────────────────────────────────────────────────────────
# NAMESPACE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

NS_CONSTITUTIONAL = "ns_constitutional"
NS_CRIMINAL       = "ns_criminal"
NS_TAX            = "ns_tax"
NS_IP             = "ns_ip"
NS_ENVIRONMENTAL  = "ns_environmental"
NS_CORPORATE      = "ns_corporate"         # Insolvency / Company Law
NS_PRIVATE        = "ns_private"           # Contract, Tort, Property, Family
NS_SPECIALIZED    = "ns_specialized"       # Catch-all for niche sub-branches

ALL_NAMESPACES = [
    NS_CONSTITUTIONAL, NS_CRIMINAL, NS_TAX, NS_IP,
    NS_ENVIRONMENTAL, NS_CORPORATE, NS_PRIVATE, NS_SPECIALIZED,
]


# ─────────────────────────────────────────────────────────────────────────────
# NAMESPACE → SUB-BRANCH MAP
# ─────────────────────────────────────────────────────────────────────────────

NAMESPACE_BRANCH_MAP: Dict[SubBranch, List[str]] = {
    SubBranch.CONSTITUTIONAL: [NS_CONSTITUTIONAL],
    SubBranch.ADMINISTRATIVE: [NS_CONSTITUTIONAL, NS_SPECIALIZED],
    SubBranch.CRIMINAL:       [NS_CRIMINAL],
    SubBranch.CONTRACT:       [NS_PRIVATE],
    SubBranch.TORT:           [NS_PRIVATE],
    SubBranch.PROPERTY:       [NS_PRIVATE],
    SubBranch.FAMILY:         [NS_PRIVATE],
    SubBranch.TAX:            [NS_TAX, NS_CONSTITUTIONAL],
    SubBranch.IP:             [NS_IP],
    SubBranch.ENVIRONMENTAL:  [NS_ENVIRONMENTAL, NS_CORPORATE],
    SubBranch.INSOLVENCY:     [NS_CORPORATE],
}


# ─────────────────────────────────────────────────────────────────────────────
# PINECONE CLIENT INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def get_pinecone_index():
    """
    Initialise Pinecone client, create index if absent, return Index object.
    Must be called once at application startup.
    """
    api_key = os.environ["PINECONE_API_KEY"]
    pc      = Pinecone(api_key=api_key)

    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name   = INDEX_NAME,
            dimension = DIMENSION,
            metric    = METRIC,
            spec      = ServerlessSpec(cloud=CLOUD, region=REGION),
        )
        # Wait for index to be ready
        import time
        while not pc.describe_index(INDEX_NAME).status.get("ready", False):
            time.sleep(1)

    return pc.Index(INDEX_NAME)


# ─────────────────────────────────────────────────────────────────────────────
# ASYNC UPSERT HELPER
# ─────────────────────────────────────────────────────────────────────────────

async def upsert_section(section: Section, namespace: str, pinecone_index) -> None:
    """
    Upsert a single Section's embedding into the specified Pinecone namespace.
    Metadata carries all retrieval-time filter fields.

    Args:
        section:        Section model with .embedding populated (1536 dims).
        namespace:      One of the NS_* constants above.
        pinecone_index: Pinecone Index object (from get_pinecone_index()).

    Raises:
        ValueError: If embedding is None or wrong dimension.
    """
    if section.embedding is None:
        raise ValueError(f"Section {section.section_id} has no embedding — run embedder first.")
    if len(section.embedding) != DIMENSION:
        raise ValueError(
            f"Expected {DIMENSION}-dim embedding, got {len(section.embedding)} "
            f"for section {section.section_id}"
        )

    vector = {
        "id":       str(section.section_id),
        "values":   section.embedding,
        "metadata": {
            "act_id":      str(section.act_id),
            "section_num": section.section_num,
            "sub_branch":  section.sub_branch.value,
            "is_repealed": section.is_repealed,
            "text_preview": section.text[:500],   # Pinecone metadata limit
        },
    }

    # Pinecone SDK is synchronous — run in executor to avoid blocking event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: pinecone_index.upsert(vectors=[vector], namespace=namespace),
    )


async def batch_upsert_sections(
    sections: list[Section],
    namespace: str,
    pinecone_index,
    batch_size: int = 100,
) -> int:
    """
    Upsert sections in batches (Pinecone limit: 100 vectors per request).
    Returns total upserted count.
    """
    total = 0
    for i in range(0, len(sections), batch_size):
        batch = sections[i : i + batch_size]
        tasks = [upsert_section(s, namespace, pinecone_index) for s in batch]
        await asyncio.gather(*tasks)
        total += len(batch)
    return total
