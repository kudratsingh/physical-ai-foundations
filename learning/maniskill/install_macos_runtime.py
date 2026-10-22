"""Unpack an official Homebrew MoltenVK bottle locally; no system installation.

Apple Silicon / macOS 14+. SAPIEN supplies the Vulkan loader; this supplies its
Metal driver. Verify the published SHA-256 before extracting. Runtime is ignored.
"""

import hashlib
import json
from pathlib import Path
import platform
import tarfile
from urllib.request import Request, urlopen

SHA256 = "69674ab4252fb0ce4c7369565105a06665af8e0c4df4c6bc251050ea012d3ece"
URL = "https://ghcr.io/v2/homebrew/core/molten-vk/blobs/sha256:" + SHA256
ROOT = Path(__file__).resolve().parents[1] / ".runtime"


def main():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("This runtime is only for Apple Silicon macOS 14+.")
    ROOT.mkdir(parents=True, exist_ok=True)
    archive = ROOT / "molten-vk-1.4.2.arm64_sonoma.bottle.tar.gz"
    if not archive.exists():
        with urlopen(
            "https://ghcr.io/token?service=ghcr.io&scope=repository:homebrew/core/molten-vk:pull",
            timeout=30,
        ) as response:
            token = json.load(response)["token"]
        with urlopen(
            Request(URL, headers={"Authorization": "Bearer " + token}), timeout=120
        ) as response:
            archive.write_bytes(response.read())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise SystemExit(
            "MoltenVK checksum mismatch; remove the downloaded archive and retry."
        )
    with tarfile.open(archive) as package:
        package.extractall(ROOT, filter="data")
    driver = next(ROOT.glob("molten-vk/*/lib/libMoltenVK.dylib"))
    (ROOT / "MoltenVK_icd.json").write_text(
        json.dumps(
            {
                "file_format_version": "1.0.0",
                "ICD": {
                    "library_path": str(driver.resolve()),
                    "api_version": "1.3.0",
                    "is_portability_driver": True,
                },
            },
            indent=2,
        )
    )
    print("Installed local driver:", driver)
    print("Set VK_ICD_FILENAMES to", ROOT / "MoltenVK_icd.json")


if __name__ == "__main__":
    main()
