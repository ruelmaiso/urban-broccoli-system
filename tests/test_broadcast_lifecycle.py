import threading
import unittest

from teacher.admin import TeacherDeployServer


class BroadcastLifecycleTests(unittest.TestCase):
    def _server_shell(self):
        server = TeacherDeployServer.__new__(TeacherDeployServer)
        server.lock = threading.Lock()
        server.broadcast_lifecycle_gate = threading.Lock()
        server.broadcast_target_ids = {"PC01"}
        server.broadcast_target_session_ids = {"PC01": 1}
        server.broadcast_session_id = 1
        server.broadcast_stop_event = threading.Event()
        server.broadcast_audio_stop_event = threading.Event()
        server.clients = {}
        server.stop_entered = threading.Event()
        def diag(event, **_kwargs):
            if event == "stop_entered":
                server.stop_entered.set()
        server._broadcast_diag = diag
        server._log_event = lambda *_args, **_kwargs: None
        server._put_bounded = lambda *_args, **_kwargs: None
        server.send_command = lambda *_args, **_kwargs: None
        return server

    def test_pre_start_stop_cannot_stop_replacement_session(self):
        """A Stop snapshot from session 1 must not remove session 2's target."""
        server = self._server_shell()
        class Gate:
            def __init__(self):
                self.lock = threading.Lock()
                self.waiting = threading.Event()

            def acquire(self, *args, **kwargs):
                return self.lock.acquire(*args, **kwargs)

            def release(self):
                self.lock.release()

            def __enter__(self):
                self.waiting.set()
                self.lock.acquire()
                return self

            def __exit__(self, *_args):
                self.lock.release()

        server.broadcast_lifecycle_gate = Gate()
        server.broadcast_lifecycle_gate.acquire()
        result = []

        worker = threading.Thread(target=lambda: result.extend(server.stop_broadcast(["PC01"])))
        worker.start()
        self.assertTrue(server.stop_entered.wait(timeout=1))
        self.assertTrue(server.broadcast_lifecycle_gate.waiting.wait(timeout=1))

        # The queued Stop has captured session 1.  Simulate Start #2 committing
        # a replacement ownership record before Stop acquires the lifecycle gate.
        with server.lock:
            server.broadcast_target_session_ids["PC01"] = 2
            server.broadcast_session_id = 2
        server.broadcast_lifecycle_gate.release()
        worker.join(timeout=1)

        self.assertFalse(worker.is_alive())
        self.assertEqual([], result)
        self.assertEqual({"PC01"}, server.broadcast_target_ids)
        self.assertEqual(2, server.broadcast_target_session_ids["PC01"])
        self.assertFalse(server.broadcast_stop_event.is_set())
        self.assertFalse(server.broadcast_audio_stop_event.is_set())


if __name__ == "__main__":
    unittest.main()
