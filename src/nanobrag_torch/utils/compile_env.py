"""The one place that reads the compile-disable environment variable.

``NANOBRAGG_DISABLE_COMPILE=1`` runs everything eagerly (needed for gradcheck and
debugging). ``NANOBRAG_DISABLE_COMPILE`` is an accepted alias, the spelling DBEX and
the runtime checklist use. Both the Simulator kernel and the ``physics`` helpers must
read the flag through here, so that either spelling turns off all compilation.
"""

import os

ENV_VARS = ("NANOBRAGG_DISABLE_COMPILE", "NANOBRAG_DISABLE_COMPILE")


def compile_disabled() -> bool:
    """True when either spelling of the compile-disable variable is set to ``"1"``."""
    return any(os.environ.get(name, "0") == "1" for name in ENV_VARS)
