# Aranya Prime Project Metadata

# Semantic Versioning: (Major, Minor, Patch, Release Level, Serial)
VERSION_INFO = (0, 1, 0, 'alpha', 1)

__version__   = "0.1.0-alpha"
__title__     = "Aranya Prime"
__author__    = "Aditya"
__email__     = "adityatheh@gmail.com"
__license__   = "Proprietary"
__copyright__ = "Copyright 2025 Aranya Research"
__status__    = "Prototype" # Prototype, Beta, Production

def get_version():
    return __version__

def get_banner():
    """Returns the ASCII banner and version info."""
    return f"""
    =========================================
      ARANYA PRIME  |  v{__version__}
    =========================================
    {__copyright__}
    Status: {__status__}
    """
