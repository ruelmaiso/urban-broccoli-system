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

    @staticmethod
    def _put_latest(target: queue.Queue[bytes], payload: bytes) -> None:
        try:
            target.put_nowait(payload)
        except queue.Full:
            try:
                target.get_nowait()
            except queue.Empty:
                pass
            try:
                target.put_nowait(payload)
            except queue.Full:
                pass

    def _normalise_pcm(self, payload: bytes, source_channels: int, source_rate: int) -> numpy.ndarray:
        """Convert a PCM16 capture chunk to this stream's stereo/chunk contract."""
        values = numpy.frombuffer(payload, dtype=numpy.int16)
        source_channels = max(1, int(source_channels))
        usable = (values.size // source_channels) * source_channels
        values = values[:usable].reshape((-1, source_channels))
        if source_channels == 1 and self.audio_channels == 2:
            values = numpy.repeat(values, 2, axis=1)
        elif source_channels != self.audio_channels:
            values = values[:, :self.audio_channels]
            if values.shape[1] < self.audio_channels:
                values = numpy.pad(values, ((0, 0), (0, self.audio_channels - values.shape[1])))
        target_frames = self.audio_chunk_frames
        if values.shape[0] != target_frames:
            if values.shape[0] == 0:
                return numpy.zeros((target_frames, self.audio_channels), dtype=numpy.int16)
            positions = numpy.linspace(0, values.shape[0] - 1, target_frames)
            source_positions = numpy.arange(values.shape[0])
            values = numpy.stack([numpy.interp(positions, source_positions, values[:, idx])
                                  for idx in range(values.shape[1])], axis=1).astype(numpy.int16)
        return values

    def _audio_producer(self, session_id: str) -> None:
        """Mix independent WASAPI loopback and default microphone capture into one PCM stream."""
        system_chunks: queue.Queue[bytes] = queue.Queue(maxsize=3)
        microphone_chunks: queue.Queue[bytes] = queue.Queue(maxsize=3)
        threading.Thread(target=self._system_audio_capture, args=(session_id, system_chunks), daemon=True,
                         name="screen-share-system-audio").start()
        threading.Thread(target=self._microphone_capture, args=(session_id, microphone_chunks), daemon=True,
                         name="screen-share-microphone").start()
        self._log("screen_share_audio_mixer_started", session_id=session_id)
        try:
            while self._session_current(session_id):
                system = microphone = None
                try:
                    system = system_chunks.get(timeout=0.03)
                except queue.Empty:
                    pass
                try:
                    microphone = microphone_chunks.get_nowait()
                except queue.Empty:
                    pass
                if system is None and microphone is None:
                    continue
                mixed = numpy.zeros((self.audio_chunk_frames, self.audio_channels), dtype=numpy.int32)
                if system is not None:
                    mixed += self._normalise_pcm(system, self.audio_channels, self.audio_rate).astype(numpy.int32)
                if microphone is not None:
                    # Keep voice intelligible without amplifying desktop playback.
                    mixed += (self._normalise_pcm(microphone, 1, self.audio_rate).astype(numpy.int32) * 7) // 10
                self._distribute("audio", numpy.clip(mixed, -32768, 32767).astype(numpy.int16).tobytes())
        finally:
            self._log("screen_share_audio_mixer_stopped", session_id=session_id)

    def _system_audio_capture(self, session_id: str, chunks: queue.Queue[bytes]) -> None:
        stream = audio = None
        self._log("screen_share_system_audio_capture_started", session_id=session_id)
        try:
            import pyaudiowpatch as pyaudio
            audio = pyaudio.PyAudio()
            loopback = audio.get_default_wasapi_loopback()
            if not loopback:
                raise RuntimeError("default WASAPI loopback device unavailable")
            if int(loopback["maxInputChannels"]) < self.audio_channels:
                raise RuntimeError("default WASAPI loopback device does not support configured channel count")
            stream = audio.open(format=pyaudio.paInt16, channels=self.audio_channels, rate=self.audio_rate,
                                input=True, input_device_index=loopback["index"], frames_per_buffer=self.audio_chunk_frames)
            while self._session_current(session_id):
                self._put_latest(chunks, stream.read(self.audio_chunk_frames, exception_on_overflow=False))
        except Exception as exc:
            self._log("screen_share_system_audio_capture_failed", session_id=session_id, reason=str(exc))
        finally:
            if stream:
                try: stream.stop_stream(); stream.close()
                except Exception: pass
            if audio:
                try: audio.terminate()
                except Exception: pass
            self._log("screen_share_system_audio_capture_stopped", session_id=session_id)

    def _microphone_capture(self, session_id: str, chunks: queue.Queue[bytes]) -> None:
        stream = audio = None
        self._log("screen_share_microphone_capture_started", session_id=session_id)
        try:
            import pyaudiowpatch as pyaudio
            audio = pyaudio.PyAudio()
            microphone = audio.get_default_input_device_info()
            name = str(microphone.get("name", "")).lower()
            if int(microphone.get("maxInputChannels", 0)) < 1 or "loopback" in name:
                raise RuntimeError("default microphone input device unavailable")
            source_rate = int(float(microphone.get("defaultSampleRate", self.audio_rate)))
            source_frames = max(1, round(source_rate * self.audio_chunk_frames / self.audio_rate))
            stream = audio.open(format=pyaudio.paInt16, channels=1, rate=source_rate, input=True,
                                input_device_index=microphone["index"], frames_per_buffer=source_frames)
            while self._session_current(session_id):
                raw = stream.read(source_frames, exception_on_overflow=False)
                self._put_latest(chunks, self._normalise_pcm(raw, 1, source_rate).tobytes())
        except Exception as exc:
            self._log("screen_share_microphone_capture_failed", session_id=session_id, reason=str(exc))
        finally:
            if stream:
                try: stream.stop_stream(); stream.close()
                except Exception: pass
            if audio:
                try: audio.terminate()
                except Exception: pass
            self._log("screen_share_microphone_capture_stopped", session_id=session_id)
