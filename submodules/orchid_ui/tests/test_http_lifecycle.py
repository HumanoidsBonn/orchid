"""Isolated real HTTP/launcher checks; no ROS or robot traffic."""
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = 'http://127.0.0.1:18000'


def request(body=None, base=BASE, path='/api/followup'):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers={'Content-Type': 'application/json'})
    try:
        response = urllib.request.urlopen(req, timeout=4)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, json.load(response)


def wait_until(fn, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            value = fn()
            if value:
                return value
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(.02)
    raise AssertionError('timed out waiting for expected HTTP/process state')


class BrokerTests(unittest.TestCase):
    def setUp(self):
        request({'cancel': True})
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def tearDown(self):
        request({'cancel': True})
        self.pool.shutdown()

    def prompt(self, text):
        future = self.pool.submit(request, {'prompt': text})
        pending = wait_until(lambda: (p if (p := request()[1]).get('prompt') == text else None))
        return future, pending['prompt_id']

    def test_stale_reply_and_acknowledgement(self):
        pending, identity = self.prompt('one')
        self.assertEqual(request({'response': 'wrong', 'prompt_id': identity + 1})[0], 409)
        self.assertEqual(request({'response': 'yes', 'prompt_id': identity}), (200, {'ok': True}))
        self.assertEqual(pending.result(), (200, {'response': 'yes'}))
        self.assertEqual(request({'response': 'duplicate', 'prompt_id': identity})[0], 409)

    def test_replacement_cleanup_and_timeout(self):
        old, old_id = self.prompt('old')
        new, new_id = self.prompt('new')
        self.assertEqual(old.result()[0], 408)
        self.assertEqual(request()[1]['prompt_id'], new_id)
        self.assertEqual(request({'cancel': True, 'prompt_id': old_id})[0], 409)
        self.assertEqual(new.result()[0], 408)
        self.assertIsNone(request()[1]['prompt'])

    def test_concurrent_cancel_reply_and_legacy_producer(self):
        pending, identity = self.prompt('race')
        a = self.pool.submit(request, {'response': 'answer', 'prompt_id': identity})
        b = self.pool.submit(request, {'cancel': True, 'prompt_id': identity})
        self.assertEqual(sorted([a.result()[0], b.result()[0]]), [200, 409])
        self.assertEqual(pending.result()[0], 200)
        pending, _ = self.prompt('legacy')
        self.assertEqual(request({'response': 'legacy answer'})[0], 200)
        self.assertEqual(pending.result()[1]['response'], 'legacy answer')


class LauncherTests(unittest.TestCase):
    def test_child_failure_cleanup_and_restart(self):
        base = 'http://127.0.0.1:18001'
        with tempfile.TemporaryDirectory(prefix='orchid-launcher-') as log_dir:
            env = dict(os.environ, ORCHID_UI_HTTP_ONLY='1', ORCHID_UI_PORT='18001', ORCHID_UI_LOG_DIR=log_dir)
            for iteration in range(2):
                with open(Path(log_dir) / f'launcher-{iteration}.log', 'w') as log:
                    proc = subprocess.Popen(['bash', str(ROOT / 'start_dashboard.sh')], cwd=ROOT, env=env, stdout=log, stderr=log)
                    try:
                        wait_until(lambda: request(base=base, path='/healthz')[0] == 200)
                        children = Path(f'/proc/{proc.pid}/task/{proc.pid}/children').read_text().split()
                        self.assertEqual(len(children), 1)
                        child = int(children[0])
                        if iteration == 0:
                            os.kill(child, signal.SIGTERM)
                            self.assertEqual(proc.wait(timeout=5), 1)
                        else:
                            proc.terminate()
                            self.assertEqual(proc.wait(timeout=5), 143)
                        self.assertFalse(Path(f'/proc/{child}').exists())
                    finally:
                        if proc.poll() is None:
                            proc.terminate()
                            proc.wait(timeout=5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
