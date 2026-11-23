"""Isolated launcher/build fixtures. No ROS nodes, network peers or robot goals."""
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


def wait_until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.05)
    raise AssertionError('launcher condition timed out')


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='orchid-launcher-check-')
        self.root = Path(self.temp.name)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        (self.root / 'ros2_ws').mkdir()
        (self.root / 'scripts').mkdir()
        for name in ('run_dashboard.sh', 'start_dashboard.sh', 'build.sh'):
            shutil.copy2(ROOT / name, self.root / name)
        (self.root / 'ros_setup.sh').write_text(':\n')
        self.processes = []
        self.logs = []
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        self.env = dict(os.environ, ORCHID_UI_ROOT=str(self.root), ORCHID_UI_LOCK_FILE=str(self.root / 'launcher.lock'),
                        ORCHID_UI_ROS_SETUP=str(self.root / 'ros_setup.sh'), ORCHID_UI_PORT=str(port),
                        ORCHID_UI_LOG_DIR=str(self.root), PATH=str(self.bin) + ':' + os.environ['PATH'])
        self.write_exec('process.py', '''import http.server, os, pathlib, signal, subprocess, sys, time
root=pathlib.Path(os.environ['ORCHID_UI_ROOT']);role=sys.argv[1]
with (root/'pids').open('a') as f:f.write(str(os.getpid())+'\\n')
if os.environ.get('IGNORE_TERM')=='1' and role=='rosapi':signal.signal(signal.SIGTERM,signal.SIG_IGN)
if role=='rosbridge':subprocess.Popen([sys.executable,__file__,'rosapi'])
if role=='web':
 class Handler(http.server.BaseHTTPRequestHandler):
  def do_GET(self):self.send_response(200);self.end_headers();self.wfile.write(b'{"ok":true}')
  def log_message(self,*args):pass
 http.server.HTTPServer(('127.0.0.1',int(os.environ['ORCHID_UI_PORT'])),Handler).serve_forever()
else:
 if os.environ.get('FAIL_CAMERA')=='1' and role=='camera':sys.exit(12)
 while True:time.sleep(.1)
''')
        self.write_exec('scripts/camera_preview.py', 'import os,runpy,sys\nsys.argv=["process.py","camera"]\nrunpy.run_path(os.environ["ORCHID_UI_ROOT"]+"/process.py",run_name="__main__")\n')
        self.write_exec('bin/ros2', f'exec {sys.executable} "$ORCHID_UI_ROOT/process.py" rosbridge\n', shell=True)
        self.write_exec('bin/python3', f'''if [[ "${{1:-}}" == -c ]]; then
 test -f "$ORCHID_UI_ROOT/interfaces.ready"
else exec {sys.executable} "$@"; fi
''', shell=True)
        self.write_exec('bin/colcon', '''echo colcon >> "$ORCHID_UI_ROOT/build.calls"
[[ "${FAIL_BUILD:-0}" != 1 ]] || exit 23
if [[ "${SLOW_BUILD:-0}" == 1 ]]; then exec "$ORCHID_UI_ROOT/process.py" build; fi
mkdir -p install
printf ':\\n' > install/setup.bash
mkdir -p install/share/ament_index/resource_index/packages
touch install/share/ament_index/resource_index/packages/orchid_ros_interfaces
if [[ "${SKIP_INTERFACES:-0}" != 1 ]]; then touch "$ORCHID_UI_ROOT/interfaces.ready"; fi
''', shell=True)
        self.write_exec('bin/rustc', 'echo "rustc 1.85.1"\n', shell=True)
        self.write_exec('bin/cargo', '''if [[ "${1:-}" == --version ]]; then echo "cargo ${FAKE_CARGO_VERSION:-1.85.1}"; exit; fi
echo cargo >> "$ORCHID_UI_ROOT/build.calls"
mkdir -p target/release
cp "$ORCHID_UI_ROOT/web.template" target/release/orchid_ui
chmod +x target/release/orchid_ui
''', shell=True)
        self.write_exec('web.template', 'import os,runpy,sys\nsys.argv=["process.py","web"]\nrunpy.run_path(os.environ["ORCHID_UI_ROOT"]+"/process.py",run_name="__main__")\n')

    def write_exec(self, name, body, shell=False):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(('#!/usr/bin/env bash\nset -e\n' if shell else f'#!{sys.executable}\n') + body)
        p.chmod(0o755)

    def ready_build(self):
        (self.root / 'ros2_ws/install').mkdir(exist_ok=True)
        (self.root / 'ros2_ws/install/setup.bash').write_text(':\n')
        marker=self.root / 'ros2_ws/install/share/ament_index/resource_index/packages/orchid_ros_interfaces'
        marker.parent.mkdir(parents=True,exist_ok=True);marker.touch()
        (self.root / 'interfaces.ready').touch()
        (self.root / 'target/release').mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.root / 'web.template', self.root / 'target/release/orchid_ui')

    def launch(self, script='run_dashboard.sh'):
        log = (self.root / f'launcher-{len(self.processes)}.log').open('w')
        self.logs.append(log)
        proc = subprocess.Popen(['bash', str(self.root / script)], env=self.env, stdout=log, stderr=log)
        self.processes.append(proc)
        return proc

    def running(self, proc):
        def ready():
            if proc.poll() is not None:
                raise AssertionError((self.root / f'launcher-{self.processes.index(proc)}.log').read_text())
            return 'Dashboard ready' in (self.root / f'launcher-{self.processes.index(proc)}.log').read_text()
        wait_until(ready)
        wait_until(lambda: (self.root / 'pids').exists() and len(self.pids()) == 4)

    def pids(self):
        p = self.root / 'pids'
        return [int(pid) for pid in p.read_text().split()] if p.exists() else []

    def stopped(self):
        wait_until(lambda: not any(alive(pid) for pid in self.pids()))

    def tearDown(self):
        for proc in self.processes:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=9)
        for pid in self.pids():
            if alive(pid):
                os.kill(pid, signal.SIGKILL)
        for log in self.logs:
            log.close()
        self.temp.cleanup()

    def test_interrupt_term_descendants_and_immediate_restart(self):
        self.ready_build()
        unrelated = subprocess.Popen(['sleep', '30'])
        try:
            for sig, result in ((signal.SIGINT, 130), (signal.SIGTERM, 143)):
                (self.root / 'pids').unlink(missing_ok=True)
                proc = self.launch()
                self.running(proc)
                proc.send_signal(sig)
                self.assertEqual(proc.wait(timeout=9), result)
                self.stopped()
                self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate(); unrelated.wait()
        self.assertFalse((self.root / 'build.calls').exists())

    def test_duplicate_does_not_stop_first_instance(self):
        self.ready_build(); first = self.launch(); self.running(first)
        duplicate = self.launch('start_dashboard.sh')
        self.assertEqual(duplicate.wait(timeout=3), 1)
        self.assertIsNone(first.poll())
        self.assertIn('already building or running', (self.root / 'launcher-1.log').read_text())

    def test_missing_workspace_builds_once(self):
        proc = self.launch(); self.running(proc)
        self.assertEqual((self.root / 'build.calls').read_text().split(), ['colcon', 'cargo'])

    def test_incomplete_interfaces_build(self):
        self.ready_build(); (self.root / 'interfaces.ready').unlink()
        proc = self.launch(); self.running(proc)
        self.assertEqual((self.root / 'build.calls').read_text().split(), ['colcon', 'cargo'])

    def test_missing_binary_build(self):
        self.ready_build(); (self.root / 'target/release/orchid_ui').unlink()
        proc = self.launch(); self.running(proc)
        self.assertTrue((self.root / 'build.calls').exists())

    def test_broken_workspace_setup_builds(self):
        self.ready_build()
        (self.root/'ros2_ws/install/setup.bash').write_text('return 1\n')
        proc=self.launch();self.running(proc)
        self.assertTrue((self.root/'build.calls').exists())

    def test_build_failure_never_starts_components(self):
        self.env['FAIL_BUILD'] = '1'
        proc = self.launch()
        self.assertEqual(proc.wait(timeout=4), 23)
        self.assertEqual(self.pids(), [])

    def test_old_cargo_fails_before_ros_build(self):
        self.env['FAKE_CARGO_VERSION']='1.75.0'
        proc=self.launch()
        self.assertEqual(proc.wait(timeout=4),1)
        self.assertFalse((self.root/'build.calls').exists())
        self.assertIn('Rust/Cargo >=1.85',(self.root/'launcher-0.log').read_text())

    def test_startup_failure_stops_other_components(self):
        self.ready_build(); self.env['FAIL_CAMERA'] = '1'
        proc = self.launch()
        self.assertEqual(proc.wait(timeout=9), 1)
        self.stopped()
        text = (self.root / 'launcher-0.log').read_text()
        self.assertIn('camera preview', text); self.assertIn('orchid_ui_camera.log', text)

    def test_component_exit_stops_group_even_without_leader(self):
        self.ready_build(); self.env['IGNORE_TERM'] = '1'
        proc = self.launch(); self.running(proc)
        # First launched role is not assumed: identify rosbridge by its child.
        bridge = next(pid for pid in self.pids() if Path(f'/proc/{pid}/task/{pid}/children').read_text().strip())
        os.kill(bridge, signal.SIGKILL)
        self.assertEqual(proc.wait(timeout=9), 1)
        self.stopped()
        self.assertIn('rosbridge/rosapi', (self.root / 'launcher-0.log').read_text())

    def test_empty_cleanup_missing_binary(self):
        proc = self.launch('start_dashboard.sh')
        self.assertEqual(proc.wait(timeout=3), 1)
        text = (self.root / 'launcher-0.log').read_text()
        self.assertIn('binary missing', text)
        self.assertNotIn('usage: kill', text)

    def test_lock_and_cancellation_during_build(self):
        self.env['SLOW_BUILD']='1'
        proc=self.launch()
        wait_until(lambda:len(self.pids())==1)
        duplicate=self.launch()
        self.assertEqual(duplicate.wait(timeout=3),1)
        proc.send_signal(signal.SIGINT)
        self.assertEqual(proc.wait(timeout=9),130)
        self.stopped()

    def test_successful_build_with_missing_interfaces_does_not_start(self):
        self.env['SKIP_INTERFACES']='1'
        proc=self.launch()
        self.assertEqual(proc.wait(timeout=4),1)
        self.assertEqual(self.pids(),[])
        self.assertIn('still unavailable after build',(self.root/'launcher-0.log').read_text())


if __name__ == '__main__':
    unittest.main(verbosity=2)
