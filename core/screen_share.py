"""LAN screen-share media distribution primitives.

The broadcaster has one capture/encode producer per active session and an
independent bounded sender queue for every student media socket.
"""
from __future__ import annotations

import queue
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import mss
import numpy

from core.protocol import recv_json_line, send_frame


@dataclass
class _Viewer:
    pc_id: str
    sock: socket.socket
    queue: queue.Queue[bytes] = field(default_factory=lambda: queue.Queue(maxsize=2))
    stop: threading.Event = field(default_factory=threading.Event)


class ScreenShareBroadcaster:
    """Teacher-side dedicated video/audio TCP services and media producers."""

    def __init__(self, bind_host: str, video_port: int, audio_port: int, *, width: int,
                 height: int, fps: int, jpeg_quality: int, audio_rate: int,
                 audio_channels: int, audio_chunk_frames: int,
                 log: Callable[..., None]) -> None:
        self.bind_host, self.video_port, self.audio_port = bind_host, video_port, audio_port
        self.width, self.height = width, height
        self.fps, self.jpeg_quality = fps, jpeg_quality
        self.audio_rate, self.audio_channels = audio_rate, audio_channels
        self.audio_chunk_frames = audio_chunk_frames
        self._log = log
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._active = threading.Event()
        self._session_id = ""
        self._video_viewers: dict[str, _Viewer] = {}
        self._audio_viewers: dict[str, _Viewer] = {}

    @property
    def active_session_id(self) -> str:
        with self._lock:
            return self._session_id if self._active.is_set() else ""

    def viewer_count(self) -> int:
        with self._lock:
            return len(set(self._video_viewers) | set(self._audio_viewers))

    def start_services(self) -> None:
        for media, port in (("video", self.video_port), ("audio", self.audio_port)):
            threading.Thread(target=self._accept_loop, args=(media, port), daemon=True,
                             name=f"screen-share-{media}-accept").start()

    def start_session(self) -> str:
        self.stop_session()
        with self._lock:
            self._session_id = uuid.uuid4().hex
            self._active.set()
            session_id = self._session_id
        self._log("SCREEN_SHARE_STARTED", session_id=session_id)
        threading.Thread(target=self._video_producer, args=(session_id,), daemon=True,
                         name="screen-share-video-capture").start()
        threading.Thread(target=self._audio_producer, args=(session_id,), daemon=True,
                         name="screen-share-audio-capture").start()
        return session_id

    def stop_session(self) -> None:
        with self._lock:
            previous = self._session_id
            self._active.clear()
            self._session_id = ""
            viewers = list(self._video_viewers.values()) + list(self._audio_viewers.values())
            self._video_viewers.clear()
            self._audio_viewers.clear()
        for viewer in viewers:
            self._close_viewer(viewer)
        if previous:
            self._log("SCREEN_SHARE_STOPPED", session_id=previous)

    def shutdown(self) -> None:
        self._stop.set()
        self.stop_session()

    def _accept_loop(self, media: str, port: int) -> None:
        while not self._stop.is_set():
            listener: Optional[socket.socket] = None
            try:
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind((self.bind_host, port))
                listener.listen()
                listener.settimeout(1.0)
                while not self._stop.is_set():
                    try:
                        sock, _ = listener.accept()
                    except socket.timeout:
                        continue
                    threading.Thread(target=self._register_viewer, args=(media, sock), daemon=True).start()
            except OSError as exc:
                if not self._stop.is_set():
                    self._log("screen_share_media_server_restart", media=media, reason=str(exc))
                    time.sleep(1)
            finally:
                if listener:
                    try:
                        listener.close()
                    except OSError:
                        pass

    def _register_viewer(self, media: str, sock: socket.socket) -> None:
        file_obj = None
        viewer: Optional[_Viewer] = None
        try:
            sock.settimeout(8.0)
            file_obj = sock.makefile("rb")
            registration = recv_json_line(file_obj)
            session_id = str((registration or {}).get("session_id", ""))
            pc_id = str((registration or {}).get("pc_id", "")).strip()
            expected_type = f"screen_share_{media}_register"
            with self._lock:
                active = self._active.is_set() and session_id == self._session_id
            if not registration or registration.get("type") != expected_type or not pc_id or not active:
                return
            sock.settimeout(None)
            viewer = _Viewer(pc_id=pc_id, sock=sock)
            viewers = self._video_viewers if media == "video" else self._audio_viewers
            with self._lock:
                old = viewers.pop(pc_id, None)
                viewers[pc_id] = viewer
            if old:
                self._close_viewer(old)
            self._log(f"screen_share_{media}_client_connected", pc_id=pc_id, session_id=session_id)
            threading.Thread(target=self._sender_loop, args=(media, viewer), daemon=True).start()
            viewer.stop.wait()
        except (OSError, ValueError) as exc:
            self._log(f"screen_share_{media}_registration_failed", reason=str(exc))
        finally:
            if file_obj:
                try: file_obj.close()
                except OSError: pass
            if viewer:
                self._remove_viewer(media, viewer)
            else:
                try: sock.close()
                except OSError: pass

    def _sender_loop(self, media: str, viewer: _Viewer) -> None:
        try:
            while not viewer.stop.is_set() and not self._stop.is_set():
                try:
                    payload = viewer.queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                send_frame(viewer.sock, payload)
        except OSError as exc:
            self._log(f"screen_share_{media}_send_failure", pc_id=viewer.pc_id, reason=str(exc))
        finally:
            viewer.stop.set()

    def _remove_viewer(self, media: str, viewer: _Viewer) -> None:
        viewers = self._video_viewers if media == "video" else self._audio_viewers
        with self._lock:
            if viewers.get(viewer.pc_id) is viewer:
                viewers.pop(viewer.pc_id, None)
        self._close_viewer(viewer)
        self._log(f"screen_share_{media}_client_disconnected", pc_id=viewer.pc_id)

    @staticmethod
    def _close_viewer(viewer: _Viewer) -> None:
        viewer.stop.set()
        try: viewer.sock.shutdown(socket.SHUT_RDWR)
        except OSError: pass
        try: viewer.sock.close()
        except OSError: pass

    def _distribute(self, media: str, payload: bytes) -> None:
        with self._lock:
            viewers = list((self._video_viewers if media == "video" else self._audio_viewers).values())
        for viewer in viewers:
            if viewer.stop.is_set():
                continue
            try:
                viewer.queue.put_nowait(payload)
            except queue.Full:
                # Keep only current media; video freshness and audio latency both matter.
                try: viewer.queue.get_nowait()
                except queue.Empty: pass
                try: viewer.queue.put_nowait(payload)
                except queue.Full: pass

    def _session_current(self, session_id: str) -> bool:
        with self._lock:
            return self._active.is_set() and self._session_id == session_id and not self._stop.is_set()

    def _video_producer(self, session_id: str) -> None:
        self._log("screen_share_capture_started", session_id=session_id)
        try:
            with mss.mss() as capture:
                monitor = capture.monitors[1]
                while self._session_current(session_id):
                    started = time.monotonic()
                    shot = capture.grab(monitor)
                    frame = cv2.cvtColor(numpy.asarray(shot), cv2.COLOR_BGRA2BGR)
                    if (frame.shape[1], frame.shape[0]) != (self.width, self.height):
                        frame = cv2.resize(frame, (self.width, self.height))
                    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                    if ok:
                        self._distribute("video", encoded.tobytes())
                    delay = (1.0 / max(1, self.fps)) - (time.monotonic() - started)
                    if delay > 0:
                        time.sleep(delay)
        except Exception as exc:
            self._log("screen_share_capture_failed", session_id=session_id, reason=str(exc))
        finally:
            self._log("screen_share_capture_stopped", session_id=session_id)

    def _audio_producer(self, session_id: str) -> None:
        """Mix WASAPI loopback and the default input into the existing PCM stream.

        The wire contract remains signed 16-bit little-endian PCM at the configured
        rate/channel count, so viewers still need only one playback stream.  Each
        source owns its PyAudio instance: a failed microphone must not take down
        loopback capture (and vice versa).
        """
        source_queues = {"system": queue.Queue(maxsize=4), "microphone": queue.Queue(maxsize=4)}
        source_done = threading.Event()
        self._log("screen_share_audio_capture_started", session_id=session_id)

        def normalize_pcm(chunk: bytes, source_rate: int, source_channels: int, state: object) -> tuple[bytes, object]:
            samples = numpy.frombuffer(chunk, dtype=numpy.int16)
            if samples.size % source_channels:
                samples = samples[:samples.size - (samples.size % source_channels)]
            samples = samples.reshape(-1, source_channels)
            if source_channels == 1 and self.audio_channels == 2:
                samples = numpy.repeat(samples, 2, axis=1)
            elif source_channels == 2 and self.audio_channels == 1:
                samples = samples.astype(numpy.int32).mean(axis=1, keepdims=True).astype(numpy.int16)
            if source_rate != self.audio_rate:
                target_frames = max(1, round(samples.shape[0] * self.audio_rate / source_rate))
                source_axis = numpy.arange(samples.shape[0])
                target_axis = numpy.linspace(0, samples.shape[0] - 1, target_frames)
                samples = numpy.stack([numpy.interp(target_axis, source_axis, samples[:, channel]) for channel in range(self.audio_channels)], axis=1).astype(numpy.int16)
            expected_samples = self.audio_chunk_frames * self.audio_channels
            flattened = samples.reshape(-1)[:expected_samples]
            if flattened.size < expected_samples:
                flattened = numpy.pad(flattened, (0, expected_samples - flattened.size))
            return flattened.astype(numpy.int16, copy=False).tobytes(), state

        def capture_source(kind: str) -> None:
            stream = audio = None
            try:
                import pyaudiowpatch as pyaudio
                audio = pyaudio.PyAudio()
                if kind == "system":
                    device = audio.get_default_wasapi_loopback()
                    if not device:
                        raise RuntimeError("default WASAPI loopback device unavailable")
                else:
                    device = audio.get_default_input_device_info()
                    if not device or device.get("isLoopbackDevice"):
                        raise RuntimeError("default microphone input device unavailable")
                source_channels = min(self.audio_channels, int(device.get("maxInputChannels", 0)))
                if source_channels <= 0:
                    raise RuntimeError("audio input device has no input channels")
                source_rate = max(8000, int(float(device.get("defaultSampleRate", self.audio_rate))))
                source_frames = max(1, round(self.audio_chunk_frames * source_rate / self.audio_rate))
                stream = audio.open(format=pyaudio.paInt16, channels=source_channels, rate=source_rate,
                                    input=True, input_device_index=device["index"], frames_per_buffer=source_frames)
                self._log("screen_share_audio_source_started", session_id=session_id, source=kind,
                          rate=source_rate, channels=source_channels, chunk_frames=source_frames)
                rate_state = None
                while self._session_current(session_id) and not source_done.is_set():
                    raw = stream.read(source_frames, exception_on_overflow=False)
                    chunk, rate_state = normalize_pcm(raw, source_rate, source_channels, rate_state)
                    try:
                        source_queues[kind].put_nowait(chunk)
                    except queue.Full:
                        try: source_queues[kind].get_nowait()
                        except queue.Empty: pass
                        try: source_queues[kind].put_nowait(chunk)
                        except queue.Full: pass
            except Exception as exc:
                self._log("screen_share_audio_source_failed", session_id=session_id, source=kind, reason=str(exc))
            finally:
                if stream:
                    try: stream.stop_stream(); stream.close()
                    except Exception: pass
                if audio:
                    try: audio.terminate()
                    except Exception: pass

        workers = [threading.Thread(target=capture_source, args=(kind,), daemon=True,
                                    name=f"screen-share-{kind}-audio") for kind in source_queues]
        for worker in workers:
            worker.start()
        expected = self.audio_chunk_frames * self.audio_channels * 2
        silence = b"\0" * expected
        latest = {"system": silence, "microphone": silence}
        try:
            while self._session_current(session_id):
                produced = False
                for kind, packets in source_queues.items():
                    try:
                        latest[kind] = packets.get(timeout=0.02 if kind == "system" else 0.0)
                        produced = True
                    except queue.Empty:
                        latest[kind] = silence
                if produced:
                    # -6 dB per source prevents clipping when both are loud.
                    system = numpy.frombuffer(latest["system"], dtype=numpy.int16).astype(numpy.int32)
                    microphone = numpy.frombuffer(latest["microphone"], dtype=numpy.int16).astype(numpy.int32)
                    mixed = numpy.clip((system + microphone) // 2, -32768, 32767).astype(numpy.int16).tobytes()
                    self._distribute("audio", mixed)
                else:
                    time.sleep(self.audio_chunk_frames / self.audio_rate)
        finally:
            source_done.set()
            self._log("screen_share_audio_capture_stopped", session_id=session_id)
