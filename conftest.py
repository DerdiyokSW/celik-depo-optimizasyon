"""Pytest kök yapılandırması — içe aktarma yolunu güvenceye alır.

Proje kökünü ``sys.path``in başına ekler ki testler invokasyon biçiminden
bağımsız olarak ``from src... import ...`` yapabilsin. Kökte ``__init__.py``
bulunmadığından pytest bunu otomatik olarak da ekler; bu dosya bu davranışı
açık ve sağlam hale getirir.
"""

import os
import sys

# Bu conftest.py proje kökünde olduğundan dizini doğrudan yola ekliyoruz.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
