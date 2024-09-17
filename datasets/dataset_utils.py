import os


def listdir(dir: str,
            endswith: str = '',
            startswith: str = '',
            return_with_dir: bool = False) -> list[str]:
    if return_with_dir:
        fnames = [os.path.join(dir, fname) for fname in os.listdir(dir)
                  if fname.endswith(endswith) and fname.startswith(startswith)]
    else:
        fnames = [fname for fname in os.listdir(dir)
                  if fname.endswith(endswith) and fname.startswith(startswith)]
    return fnames