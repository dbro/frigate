"""Model identity: the MD5 of the model file, like Frigate's detector model_hash.

Secondlook stores the MD5 next to the file's basename (a readable label), so a
replaced file under the same name is told apart. Computed once per process at
startup; about 0.1-0.3 s for a 100 MB file.
"""

from __future__ import annotations

import hashlib
import os


def file_md5(path: str | None) -> str | None:
    """Hex MD5 of the file (8 KiB reads, as detector_config.compute_model_hash).

    None when there is no such file: unlike the detector's md5(b"unknown"),
    a missing file is not given an id that looks like a real one.
    """
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            digest.update(chunk)
    return digest.hexdigest()
