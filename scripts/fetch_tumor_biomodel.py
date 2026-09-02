"""Fetch + cache BioModels BIOMD0000000903 SBML into workspace/datasets/.

Run once; the cached XML is committed so composite runs are reproducible/offline.
"""
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

from pbg_biomodels.run_biomodels import load_biomodel

DEST = Path(__file__).resolve().parents[1] / "workspace" / "datasets" / "BIOMD0000000903.xml"

# Direct OMEX archive URL (redirects biomodels.org → CloudFront).  The archive
# contains the SBML as Solis-perez2019.xml; no .sedml is needed for this path.
_OMEX_URL = (
    "https://www.biomodels.org/biomodels/services/download/get-files/"
    "MODEL1912180005/3/MODEL1912180005.3.omex"
)
_SBML_IN_OMEX = "Solis-perez2019.xml"


def _download_from_omex() -> None:
    """Download the OMEX archive and extract the SBML file."""
    print(f"Downloading OMEX from {_OMEX_URL} ...")
    with urllib.request.urlopen(_OMEX_URL) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        with zf.open(_SBML_IN_OMEX) as sbml_src, DEST.open("wb") as dst:
            shutil.copyfileobj(sbml_src, dst)


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    # Primary: use pbg_biomodels loader (populates local cache + metadata).
    try:
        result = load_biomodel("BIOMD0000000903", None)
        src = Path(result.sbml_path)
        shutil.copyfile(src, DEST)
    except Exception as exc:
        # Fallback: BIOMD0000000903 has no .sedml file so load_biomodel raises;
        # download the SBML directly from the BioModels OMEX archive instead.
        print(f"load_biomodel raised {type(exc).__name__}: {exc} — falling back to OMEX download")
        _download_from_omex()
    print(f"cached -> {DEST} ({DEST.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
