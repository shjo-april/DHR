import sys

if sys.platform == 'linux':
    from .densecrf import DenseCRF