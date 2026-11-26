"""Run a real OCR scan with Python socket and DNS activity disabled."""
from pathlib import Path
import runpy
import sys


program = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(program.parent))
sys.argv = [str(program), *sys.argv[2:]]
attempts = []


def no_network(event, arguments):
    if event in {'socket.connect', 'socket.getaddrinfo', 'socket.gethostbyname', 'socket.sendto'}:
        attempts.append(event)
        raise RuntimeError('Network activity is disabled for this QC run.')


sys.addaudithook(no_network)
runpy.run_path(str(program), run_name='__main__')
assert not attempts, attempts
print('OCR scan completed with zero Python socket or DNS attempts.')
