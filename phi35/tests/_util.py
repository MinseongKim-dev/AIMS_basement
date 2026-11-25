"""Shared helpers for the plain-script tests (pytest is not required)."""
import contextlib
import sys
from pathlib import Path

PHI35 = Path(__file__).resolve().parents[1]
REPO = PHI35.parent
for p in (str(PHI35), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


@contextlib.contextmanager
def raises(*types, match=None):
    try:
        yield
    except types as e:
        if match is not None:
            assert match in str(e), f'{match!r} not in {e}'
        return
    raise AssertionError(f'did not raise {[t.__name__ for t in types]}')


def run(namespace):
    tests = [(n, f) for n, f in namespace.items() if n.startswith('test_') and callable(f)]
    for name, f in tests:
        f()
        print('ok', name)
    print(f'{len(tests)} tests passed')
