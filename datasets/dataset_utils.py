import os

def listdir(dir: str, endswith: str) -> list[str]:
    return [fname for fname in os.listdir(dir) if fname.endswith(endswith)]