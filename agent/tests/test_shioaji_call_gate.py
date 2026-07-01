"""_ShioajiCallGate: thread-safety (mutual exclusion) + rate limiting.

Built alongside parallelizing the per-code fetch loop in shioaji_loader.py /
shioaji_futures_loader.py. suppress_native_stdout()'s os.dup2 redirects fd 1
for the whole *process*, not per-thread -- concurrently entering/exiting that
context from multiple threads would race on the same fd, and a torn
dup2/close sequence can leave fd 1 permanently broken (fatal for the MCP
stdio transport, which depends on that exact fd). The gate is the single
choke point that makes concurrent loader fetches safe: only one thread is
ever mid-native-call at a time, and Shioaji's documented 50-queries/10s
quota (shared account-wide, not per-thread) is enforced at the same point.
"""

from __future__ import annotations

import threading
import time

import pytest

from backtest.loaders._shioaji_kbars import _ShioajiCallGate


class TestMutualExclusion:
    def test_never_more_than_one_thread_inside_at_once(self) -> None:
        gate = _ShioajiCallGate(max_calls=1000, period_seconds=60.0)
        concurrent = 0
        max_concurrent_seen = 0
        lock = threading.Lock()

        def worker() -> None:
            nonlocal concurrent, max_concurrent_seen
            with gate.call():
                with lock:
                    concurrent += 1
                    max_concurrent_seen = max(max_concurrent_seen, concurrent)
                time.sleep(0.02)  # hold the gate long enough for overlap to be likely if unsafe
                with lock:
                    concurrent -= 1

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert max_concurrent_seen == 1

    def test_all_calls_eventually_complete(self) -> None:
        """No deadlock, no lost calls -- every submitted call runs exactly once."""
        gate = _ShioajiCallGate(max_calls=1000, period_seconds=60.0)
        completed: list[int] = []
        lock = threading.Lock()

        def worker(i: int) -> None:
            with gate.call():
                with lock:
                    completed.append(i)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert sorted(completed) == list(range(30))


class TestRateLimiting:
    def test_calls_within_limit_do_not_wait(self) -> None:
        gate = _ShioajiCallGate(max_calls=5, period_seconds=10.0)
        start = time.monotonic()
        for _ in range(5):
            with gate.call():
                pass
        elapsed = time.monotonic() - start
        assert elapsed < 0.5  # well under the 10s window -- no throttling needed

    def test_exceeding_limit_forces_a_wait(self) -> None:
        gate = _ShioajiCallGate(max_calls=3, period_seconds=0.3)
        start = time.monotonic()
        for _ in range(6):  # 2x the limit
            with gate.call():
                pass
        elapsed = time.monotonic() - start
        # 6 calls at 3-per-0.3s must take at least one full extra window.
        assert elapsed >= 0.25

    def test_old_calls_age_out_of_the_window(self) -> None:
        gate = _ShioajiCallGate(max_calls=2, period_seconds=0.2)
        with gate.call():
            pass
        with gate.call():
            pass
        time.sleep(0.25)  # let both calls age out of the 0.2s window
        start = time.monotonic()
        with gate.call():
            pass
        elapsed = time.monotonic() - start
        assert elapsed < 0.1  # slot was free -- no wait needed

    def test_concurrent_threads_share_one_counter(self) -> None:
        """The rate limit is account-wide -- concurrent threads must not each
        get their own independent budget."""
        gate = _ShioajiCallGate(max_calls=4, period_seconds=1.0)
        call_times: list[float] = []
        lock = threading.Lock()

        def worker() -> None:
            with gate.call():
                with lock:
                    call_times.append(time.monotonic())

        threads = [threading.Thread(target=worker) for _ in range(8)]
        start = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # 8 calls at a shared 4-per-1s budget must take at least ~1s total,
        # not ~0s (which is what 8 independent per-thread budgets would give).
        assert max(call_times) - start >= 0.9


class TestGateSuppressesNativeStdout:
    def test_stdout_is_redirected_during_the_call(self) -> None:
        """The gate must still provide suppress_native_stdout's fd redirect,
        not just the lock -- both concerns collapse into this one gate."""
        import os

        gate = _ShioajiCallGate(max_calls=10, period_seconds=10.0)
        saved_fd = os.dup(1)
        try:
            with gate.call():
                # Writing directly to fd 1 (bypassing sys.stdout) must be
                # redirected away, same guarantee suppress_native_stdout gives.
                inside_target = os.readlink(f"/proc/self/fd/1")
            outside_target = os.readlink(f"/proc/self/fd/1")
        finally:
            os.dup2(saved_fd, 1)
            os.close(saved_fd)
        assert "null" in inside_target
        assert "null" not in outside_target
