"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "battlefield_stkg_dataset"
ONTOLOGY_DIR = DATA_ROOT / "ontology"


@pytest.fixture(scope="session")
def data_root() -> Path:
    return DATA_ROOT


@pytest.fixture(scope="session")
def ontology_dir() -> Path:
    return ONTOLOGY_DIR


@pytest.fixture(scope="session")
def ontology():
    from citanet.data.ontology import load_ontology
    return load_ontology(ONTOLOGY_DIR)


@pytest.fixture()
def scn_0001():
    from citanet.data.loader import load_scenario
    return load_scenario(DATA_ROOT / "scenarios" / "scn_0001", ONTOLOGY_DIR)
