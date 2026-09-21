import csv
import json
import logging
import queue
import socket
import threading
import time
import traceback
import uuid
import tkinter as tk
from tkinter import messagebox
from collections import deque
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import customtkinter as ctk
import mss
import numpy
import sounddevice as sd
from PIL import Image, ImageDraw, ImageFont, ImageTk

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from config.deploy_settings import NETWORK, RUNTIME
from core.auth_db import AuthDatabase
from core.client_registry import ClientRegistry
from core.heartbeat import HeartbeatState
from core.protocol import recv_frame, recv_json_line, send_frame, send_json
from core.app_settings import AppSettings, SettingsStore
from teacher.reservations import ReservationManager
from teacher.ui.student_management import StudentManagementPanel
from teacher.ui.reservation_management import ReservationManagementPanel
from teacher.theme import (
    STATUS_COLORS as THEME_STATUS_COLORS,
    ESSU_PRIMARY,
    ESSU_WARNING,
    ESSU_ERROR,
    UI_BG,
    CARD_BG,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    BORDER_SUBTLE,
    BUTTON_PRIMARY,
    BUTTON_NEUTRAL,
    BUTTON_WARNING,
    DARK_TEXT_SECONDARY,
)


@dataclass
class ClientState:
    pc_id: str
    mac: str
    hostname: str
    control_sock: Optional[socket.socket] = None
    control_send_lock: threading.Lock = field(default_factory=threading.Lock)
    video_sock: Optional[socket.socket] = None
    last_frame: Optional[Image.Image] = None
    heartbeat: HeartbeatState = field(default_factory=HeartbeatState)
    online: bool = False
    locked: bool = False
    last_ack: str = ""
    last_ack_result: str = ""
    peer_ip: str = ""
    cpu_percent: Optional[float] = None
    ram_percent: Optional[float] = None
    disk_percent: Optional[float] = None
    uptime_s: Optional[int] = None
    current_user: str = ""
    year_section: str = ""
    student_number: str = ""
    auth_session_id: Optional[int] = None
    session_timer: Optional["TimerState"] = None
    interrupted: bool = False
    interrupted_remaining_s: int = 0
    interrupted_until_ts: Optional[float] = None
    interrupted_student_number: str = ""
    control_generation: int = 0
    video_generation: int = 0
    heartbeat_generation: int = 0
    lock_intent: bool = False
    lock_reason: Optional[str] = None


@dataclass
class SensorState:
    temperature: Optional[float] = None
    fan_ok: Optional[bool] = None
    fan_rpm: Optional[int] = None
    last_update: float = 0.0


@dataclass
class TimerState:
    timer_id: str
    targets: list[str]
    start_ts: float
    duration_ms: int
    warning_ms: int
    active: bool = True
    paused: bool = False
    paused_at_ts: Optional[float] = None
    paused_accum_ms: int = 0


@dataclass
class ClientMetrics:
    reconnect_count: int = 0
    dropped_frame_events: int = 0
    invalid_messages: int = 0
    stale_sensor_transitions: int = 0


@dataclass
class EffectiveState:
    pc_id: str
    status_text: str = "OFFLINE (NO HEARTBEAT)"
    color: str = "#8b1e1e"
    is_online: bool = False
    sensor_age_s: Optional[int] = None
    updated_ts: float = 0.0


@dataclass
class ErrorEvent:
    code: str
    pc_id: str
    detail: str
    ts: float = field(default_factory=time.time)


@dataclass
class RecordingState:
    recording_id: int
    session_id: int
    pc_id: str
    file_path: str
    writer: cv2.VideoWriter
    started_ts: float


@dataclass
class HealthSnapshot:
    ts: float
    total_clients: int
    online_clients: int
    healthy_clients: int
    degraded_clients: int
    stale_clients: int
    waiting_sensor_clients: int
    locked_clients: int
    invalid_messages: int
    dropped_frames: int


STATUS_COLORS = THEME_STATUS_COLORS

CONVERSATION_TTL_S = 6 * 60 * 60
CONVERSATION_MAX_PCS = 500

VALID_COMMAND_ACKS = {"LOCK_NOW", "UNLOCK_NOW", "SET_TIMER", "EXTEND_TIMER", "CANCEL_TIMER", "TIMER_EXPIRED", "TIMER_WARNING", "SET_STREAM_PROFILE", "SET_RUNTIME_TUNING", "SESSION_MESSAGE", "EXTENSION_REQUEST", "EXTENSION_OFFER", "PAUSE_TIMER", "RESUME_TIMER", "SCREEN_SHARE_START", "SCREEN_SHARE_STOP", "SHUTDOWN", "RESTART"}
ERROR_CODES = {
    "CONTROL_VALIDATION_ERROR",
    "SENSOR_VALIDATION_ERROR",
    "SENSOR_OUTLIER",
    "VIDEO_DECODE_ERROR",
}

LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "teacher_runtime.log"
HEALTH_FILE = LOG_DIR / "health_latest.json"
HEALTH_HISTORY_LIMIT = 720


def _write_json_atomic(path: Path, payload: object) -> None:
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


class TeacherDeployServer:
    def __init__(self) -> None:
        self.registry = ClientRegistry(ROOT / "client_registry.json")
        self.auth_db = AuthDatabase(ROOT / "data" / "lab_monitor.db")
        self.reservation_manager = ReservationManager(self.auth_db)
        self.clients: dict[str, ClientState] = {}
        self.sensors: dict[str, SensorState] = {}
        self.timers: dict[str, TimerState] = {}
        self.effective_states: dict[str, EffectiveState] = {}
        self.metrics: dict[str, ClientMetrics] = {}
        for pc_id in self.registry.list_pc_ids():
            self.sensors[pc_id] = SensorState()
            self.metrics[pc_id] = ClientMetrics()
        self.error_events: deque[ErrorEvent] = deque(maxlen=256)
        self.health_history: deque[HealthSnapshot] = deque(maxlen=HEALTH_HISTORY_LIMIT)
        self.frame_queue: queue.Queue[tuple[str, Image.Image]] = queue.Queue(maxsize=RUNTIME.frame_queue_max)
        self.status_queue: queue.Queue[str] = queue.Queue(maxsize=64)
        self.sensor_queue: queue.Queue[str] = queue.Queue(maxsize=64)
        self.lock = threading.Lock()
        self.logger = self._create_logger()
        self.settings_store = SettingsStore(ROOT / "data" / "app_settings.json")
        self.settings = self.settings_store.load()
        self.default_session_s = int(self.settings.session_duration_s)
        self.recordings: dict[str, RecordingState] = {}
        self.active_sessions_by_pc_id: dict[str, dict] = {}
        self.pending_signout_lock_pc_ids: set[str] = set()
        self.message_events: deque[dict] = deque(maxlen=1000)
        self.conversations: dict[str, dict] = {}
        self.extension_requests: dict[str, dict] = {}
        self.extension_status_by_pc: dict[str, str] = {}
        self.session_extension_ms_by_timer_pc: dict[tuple[str, str], int] = {}
        self.video_status_last_ts_by_pc: dict[str, float] = {}
        # Screen-share media is deliberately separate from student monitoring video.
        self.screen_share_session_id: Optional[str] = None
        self.screen_share_targets: set[str] = set()
        self.screen_share_timer_paused_by_us: set[str] = set()
        self.screen_share_stop_event = threading.Event()
        self.screen_share_video_queues: dict[str, queue.Queue[bytes]] = {}
        self.screen_share_audio_queues: dict[str, queue.Queue[bytes]] = {}
        self.udp_send_sock: Optional[socket.socket] = None
        self.udp_send_lock = threading.Lock()
        self.timers_file = ROOT / "data" / "active_timers.json"
        self.session_timers_file = ROOT / "data" / "active_session_timers.json"
        self.auth_db.close_all_active_recordings(status="server_restart")
        self._load_active_sessions_from_db()
        persisted_session_timers = self._load_session_timers()
        if persisted_session_timers:
            for pc_id, payload in persisted_session_timers.items():
                if pc_id in self.active_sessions_by_pc_id:
                    timer = self._session_timer_from_payload(pc_id, payload)
                    if timer:
                        self.active_sessions_by_pc_id[pc_id]["session_timer"] = payload
                        client_ref = self.clients.get(pc_id)
                        if client_ref:
                            client_ref.session_timer = timer
        self._load_persisted_timers()

    def _load_active_sessions_from_db(self) -> None:
        now = time.time()
        restored: dict[str, dict] = {}
        for row in self.auth_db.get_active_sessions():
            pc_id = str(row.get("pc_id", "")).strip()
            if not pc_id:
                continue
            login_ts = float(row.get("login_ts", now))
            session_timer = self._new_session_timer(
                pc_id=pc_id,
                duration_ms=int(self.settings.session_duration_s) * 1000,
                start_ts=login_ts,
            )
            if self._timer_remaining_ms(session_timer, now) <= 0:
                self.auth_db.close_active_session(pc_id, status="session_timeout")
                self.pending_signout_lock_pc_ids.add(pc_id)
                continue
            restored[pc_id] = {
                "session_id": int(row.get("session_id", 0)),
                "full_name": str(row.get("full_name", "")),
                "student_number": str(row.get("student_number", "")),
                "year_section": str(row.get("year_section", "")),
                "login_ts": login_ts,
                "session_timer": self._session_timer_payload(session_timer),
            }
        self.active_sessions_by_pc_id = restored

    def _new_session_timer(self, pc_id: str, duration_ms: int, start_ts: Optional[float] = None) -> TimerState:
        return TimerState(
            timer_id=f"session-{pc_id}",
            targets=[pc_id],
            start_ts=time.time() if start_ts is None else float(start_ts),
            duration_ms=max(0, int(duration_ms)),
            warning_ms=0,
            active=True,
            paused=False,
            paused_at_ts=None,
            paused_accum_ms=0,
        )

    def _session_timer_payload(self, timer: Optional[TimerState]) -> Optional[dict]:
        if timer is None:
            return None
        return {
            "start_ts": float(timer.start_ts),
            "duration_ms": int(timer.duration_ms),
            "paused": bool(timer.paused),
            "paused_at_ts": float(timer.paused_at_ts) if timer.paused_at_ts is not None else None,
            "paused_accum_ms": int(timer.paused_accum_ms),
        }

    def _session_timer_from_payload(self, pc_id: str, payload: object) -> Optional[TimerState]:
        if not isinstance(payload, dict):
            return None
        try:
            timer = self._new_session_timer(
                pc_id=pc_id,
                duration_ms=int(payload.get("duration_ms", 0)),
                start_ts=float(payload.get("start_ts", time.time())),
            )
            timer.paused = bool(payload.get("paused", False))
            timer.paused_at_ts = float(payload.get("paused_at_ts")) if payload.get("paused_at_ts") is not None else None
            timer.paused_accum_ms = max(0, int(payload.get("paused_accum_ms", 0)))
            return timer
        except Exception:
            return None

    def _persist_session_timers(self) -> None:
        try:
            with self.lock:
                payload: dict[str, dict] = {}
                for pc_id, entry in self.active_sessions_by_pc_id.items():
                    if not isinstance(entry, dict):
                        continue
                    client_ref = self.clients.get(pc_id)
                    timer_state = client_ref.session_timer if client_ref else None
                    if timer_state:
                        timer_payload = self._session_timer_payload(timer_state)
                    else:
                        timer_payload = entry.get("session_timer") if isinstance(entry.get("session_timer"), dict) else None
                    if timer_payload:
                        payload[pc_id] = timer_payload
            temp_path = self.session_timers_file.with_suffix(".json.tmp")
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temp_path.replace(self.session_timers_file)
        except Exception as exc:
            self._log_event("session_timers_persist_error", reason=str(exc))

    def _load_session_timers(self) -> dict[str, dict]:
        if not self.session_timers_file.exists():
            return {}
        try:
            data = json.loads(self.session_timers_file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return {}
            return {str(k): v for k, v in data.items() if isinstance(v, dict)}
        except Exception:
            return {}

    def _persist_timers(self) -> None:
        payload = []
        with self.lock:
            timers = list(self.timers.values())
        for timer in timers:
            payload.append({
                "timer_id": timer.timer_id,
                "targets": timer.targets,
                "start_ts": timer.start_ts,
                "duration_ms": timer.duration_ms,
                "warning_ms": timer.warning_ms,
                "active": bool(timer.active),
                "paused": bool(timer.paused),
                "paused_at_ts": timer.paused_at_ts,
                "paused_accum_ms": int(timer.paused_accum_ms),
            })
        try:
            _write_json_atomic(self.timers_file, payload)
        except Exception as exc:
            self._log_event("timers_persist_error", reason=str(exc))

    def _load_persisted_timers(self) -> None:
        if not self.timers_file.exists():
            return
        try:
            payload = json.loads(self.timers_file.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, list):
            return
        loaded: dict[str, TimerState] = {}
        for row in payload:
            if not isinstance(row, dict):
                continue
            timer_id = str(row.get("timer_id", "")).strip()
            targets = row.get("targets", [])
            if not timer_id or not isinstance(targets, list):
                continue
            try:
                loaded[timer_id] = TimerState(
                    timer_id=timer_id,
                    targets=[str(v) for v in targets],
                    start_ts=float(row.get("start_ts", time.time())),
                    duration_ms=int(row.get("duration_ms", 0)),
                    warning_ms=int(row.get("warning_ms", 0)),
                    active=bool(row.get("active", True)),
                    paused=bool(row.get("paused", False)),
                    paused_at_ts=float(row.get("paused_at_ts")) if row.get("paused_at_ts") is not None else None,
                    paused_accum_ms=int(row.get("paused_accum_ms", 0)),
                )
            except Exception:
                continue
        self.timers = loaded

    def _put_bounded(self, q: queue.Queue, item) -> None:
        try:
            q.put_nowait(item)
        except queue.Full:
            if q is self.frame_queue and isinstance(item, tuple) and item:
                self._metrics_for(item[0]).dropped_frame_events += 1
            if self.settings.frame_queue_policy == "drop_newest":
                return
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            q.put_nowait(item)

    def _create_logger(self) -> logging.Logger:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("teacher-runtime")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
        logger.propagate = False
        return logger

    def _log_event(self, event: str, **data: object) -> None:
        payload = {"ts": round(time.time(), 3), "event": event, **data}
        self.logger.info(json.dumps(payload, sort_keys=True))

    def _run_guarded_loop(self, loop_name: str, target, restart_delay_s: float = 1.0) -> None:
        while True:
            try:
                target()
            except Exception as exc:
                self._log_event("background_loop_restart", loop=loop_name, reason=str(exc))
                time.sleep(max(0.2, float(restart_delay_s)))

    def _aggregate_health(self) -> HealthSnapshot:
        with self.lock:
            total_clients = len(self.clients)
            online_clients = sum(1 for s in self.effective_states.values() if s.is_online)
            healthy_clients = sum(1 for s in self.effective_states.values() if s.status_text == "ONLINE (HEALTHY)")
            degraded_clients = sum(1 for s in self.effective_states.values() if s.status_text == "ONLINE (DEGRADED: VIDEO DELAY)")
            stale_clients = sum(1 for s in self.effective_states.values() if s.status_text == "ONLINE (SENSOR STALE)")
            waiting_sensor_clients = sum(1 for s in self.effective_states.values() if s.status_text == "ONLINE (SENSOR DATA WAITING)")
            locked_clients = sum(1 for s in self.effective_states.values() if s.status_text == "LOCKED (MANUAL/TIMER)")
            invalid_messages = sum(m.invalid_messages for m in self.metrics.values())
            dropped_frames = sum(m.dropped_frame_events for m in self.metrics.values())
        return HealthSnapshot(
            ts=time.time(),
            total_clients=total_clients,
            online_clients=online_clients,
            healthy_clients=healthy_clients,
            degraded_clients=degraded_clients,
            stale_clients=stale_clients,
            waiting_sensor_clients=waiting_sensor_clients,
            locked_clients=locked_clients,
            invalid_messages=invalid_messages,
            dropped_frames=dropped_frames,
        )

    def _write_health_snapshot(self, snapshot: HealthSnapshot) -> None:
        payload = {
            "ts": round(snapshot.ts, 3),
            "total_clients": snapshot.total_clients,
            "online_clients": snapshot.online_clients,
            "healthy_clients": snapshot.healthy_clients,
            "degraded_clients": snapshot.degraded_clients,
            "stale_clients": snapshot.stale_clients,
            "waiting_sensor_clients": snapshot.waiting_sensor_clients,
            "locked_clients": snapshot.locked_clients,
            "invalid_messages": snapshot.invalid_messages,
            "dropped_frames": snapshot.dropped_frames,
        }
        try:
            _write_json_atomic(HEALTH_FILE, payload)
        except Exception as exc:
            self._log_event("health_snapshot_write_error", reason=str(exc))

    def _metrics_for(self, pc_id: str) -> ClientMetrics:
        if pc_id not in self.metrics:
            self.metrics[pc_id] = ClientMetrics()
        return self.metrics[pc_id]

    def _record_error(self, code: str, pc_id: str, detail: str) -> None:
        error_code = code if code in ERROR_CODES else "CONTROL_VALIDATION_ERROR"
        event = ErrorEvent(code=error_code, pc_id=pc_id or "unknown", detail=detail)
        with self.lock:
            self.error_events.append(event)
            if pc_id:
                self._metrics_for(pc_id).invalid_messages += 1
        self._log_event("error", code=error_code, pc_id=event.pc_id, detail=detail)

    def _lock_priority(self, reason: Optional[str]) -> int:
        order = {None: 0, "timer": 1, "manual": 2, "signout": 3}
        return order.get(reason, 0)

    def _apply_lock_reason(self, client: ClientState, reason: Optional[str], clear: bool = False) -> None:
        if clear:
            if client.lock_reason == "signout":
                client.lock_intent = True
                client.locked = True
                return
            client.lock_reason = None
            client.lock_intent = False
            client.locked = False
            return
        if self._lock_priority(reason) >= self._lock_priority(client.lock_reason):
            client.lock_reason = reason
            client.lock_intent = bool(reason)
            client.locked = bool(reason)

    def _clear_signout_lock_after_auth(self, client: ClientState, pc_id: str) -> None:
        """Clear signout lock state once authentication succeeds on this workstation."""
        if client.lock_reason == "signout":
            client.lock_reason = None
            client.lock_intent = False
            client.locked = False
        self.pending_signout_lock_pc_ids.discard(pc_id)

    def _status_for(self, pc_id: str, now: float) -> EffectiveState:
        client = self.clients.get(pc_id)
        sensor = self.sensors.get(pc_id, SensorState())
        if not client or not client.online:
            return EffectiveState(pc_id=pc_id, status_text="OFFLINE (NO HEARTBEAT)", color=STATUS_COLORS["OFFLINE (NO HEARTBEAT)"], is_online=False, updated_ts=now)
        if client.lock_intent:
            return EffectiveState(pc_id=pc_id, status_text="LOCKED (MANUAL/TIMER)", color=STATUS_COLORS["LOCKED (MANUAL/TIMER)"], is_online=True, updated_ts=now)
        if sensor.last_update <= 0:
            return EffectiveState(pc_id=pc_id, status_text="ONLINE (SENSOR DATA WAITING)", color=STATUS_COLORS["ONLINE (SENSOR DATA WAITING)"], is_online=True, updated_ts=now)
        sensor_age = int(max(0, now - sensor.last_update))
        if sensor_age > RUNTIME.sensor_timeout_s:
            return EffectiveState(pc_id=pc_id, status_text="ONLINE (SENSOR STALE)", color=STATUS_COLORS["ONLINE (SENSOR STALE)"], is_online=True, sensor_age_s=sensor_age, updated_ts=now)
        if client.last_frame is None:
            return EffectiveState(pc_id=pc_id, status_text="ONLINE (DEGRADED: VIDEO DELAY)", color=STATUS_COLORS["ONLINE (DEGRADED: VIDEO DELAY)"], is_online=True, sensor_age_s=sensor_age, updated_ts=now)
        return EffectiveState(pc_id=pc_id, status_text="ONLINE (HEALTHY)", color=STATUS_COLORS["ONLINE (HEALTHY)"], is_online=True, sensor_age_s=sensor_age, updated_ts=now)

    def _run_reconciliation_loop(self) -> None:
        while True:
            now = time.time()
            changed_ids: list[str] = []
            with self.lock:
                for pc_id in list(self.clients.keys()):
                    previous = self.effective_states.get(pc_id)
                    current = self._status_for(pc_id, now)
                    if previous and previous.status_text != "ONLINE (SENSOR STALE)" and current.status_text == "ONLINE (SENSOR STALE)":
                        self._metrics_for(pc_id).stale_sensor_transitions += 1
                    self.effective_states[pc_id] = current
                    if not previous or previous.status_text != current.status_text:
                        changed_ids.append(pc_id)
                        self._log_event(
                            "status_transition",
                            pc_id=pc_id,
                            previous_status=previous.status_text if previous else "UNKNOWN",
                            current_status=current.status_text,
                        )
            for pc_id in changed_ids:
                self._put_bounded(self.status_queue, pc_id)
            time.sleep(0.5)

    def _run_observability_loop(self) -> None:
        while True:
            snapshot = self._aggregate_health()
            with self.lock:
                self.health_history.append(snapshot)
            self._write_health_snapshot(snapshot)
            self._log_event(
                "health_snapshot",
                total_clients=snapshot.total_clients,
                online_clients=snapshot.online_clients,
                healthy_clients=snapshot.healthy_clients,
                degraded_clients=snapshot.degraded_clients,
                stale_clients=snapshot.stale_clients,
                waiting_sensor_clients=snapshot.waiting_sensor_clients,
                locked_clients=snapshot.locked_clients,
                invalid_messages=snapshot.invalid_messages,
                dropped_frames=snapshot.dropped_frames,
            )
            time.sleep(5)

    def start(self) -> None:
        threading.Thread(target=self._run_control_server, daemon=True).start()
        threading.Thread(target=self._run_video_server, daemon=True).start()
        threading.Thread(target=self._run_sensor_server, daemon=True).start()
        threading.Thread(target=self._run_udp_fallback_server, daemon=True).start()
        threading.Thread(target=self._run_screen_share_media_server, args=(NETWORK.screen_share_video_port, "video"), daemon=True).start()
        threading.Thread(target=self._run_screen_share_media_server, args=(NETWORK.screen_share_audio_port, "audio"), daemon=True).start()
        threading.Thread(target=self._run_guarded_loop, args=("heartbeat_monitor", self._monitor_heartbeats), daemon=True).start()
        threading.Thread(target=self._run_guarded_loop, args=("reconciliation", self._run_reconciliation_loop), daemon=True).start()
        threading.Thread(target=self._run_guarded_loop, args=("observability", self._run_observability_loop), daemon=True).start()
        threading.Thread(target=self._run_guarded_loop, args=("student_sessions", self._monitor_student_sessions), daemon=True).start()

    def _run_control_server(self) -> None:
        while True:
            sock: Optional[socket.socket] = None
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((self.settings.teacher_bind_host, NETWORK.control_port))
                sock.listen()
                while True:
                    client_sock, addr = sock.accept()
                    peer_ip = str(addr[0]) if addr else ""
                    threading.Thread(target=self._control_client_loop, args=(client_sock, peer_ip), daemon=True).start()
            except OSError as exc:
                self._log_event("control_server_restart", reason=str(exc))
                time.sleep(1)
            finally:
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass

    def _control_client_loop(self, client_sock: socket.socket, peer_ip: str) -> None:
        pc_id: Optional[str] = None
        owned_generation = 0
        file_obj = None
        try:
            file_obj = client_sock.makefile("rb")
            while True:
                if pc_id:
                    with self.lock:
                        current = self.clients.get(pc_id)
                        if not current or current.control_generation != owned_generation or current.control_sock is not client_sock:
                            break
                msg = recv_json_line(file_obj)
                if msg is None:
                    break
                msg_type = msg.get("type")
                if msg_type == "register":
                    mac = str(msg.get("mac", "")).strip().lower()
                    hostname = str(msg.get("hostname", "")).strip() or "unknown"
                    if not mac:
                        self._record_error("CONTROL_VALIDATION_ERROR", "", "register missing mac")
                        send_json(client_sock, {"type": "error", "reason": "missing mac"})
                        continue
                    assigned = self.registry.get_or_assign_id(mac, hostname)
                    with self.lock:
                        if assigned not in self.clients:
                            self.clients[assigned] = ClientState(pc_id=assigned, mac=mac, hostname=hostname)
                            self.sensors[assigned] = SensorState()
                            self.metrics[assigned] = ClientMetrics()
                        else:
                            self._metrics_for(assigned).reconnect_count += 1
                        state = self.clients[assigned]
                        state.mac = mac
                        state.hostname = hostname
                        if state.control_sock and state.control_sock is not client_sock:
                            try:
                                state.control_sock.close()
                            except OSError:
                                pass
                        state.control_generation += 1
                        owned_generation = state.control_generation
                        state.control_sock = client_sock
                        state.peer_ip = peer_ip
                        state.heartbeat.touch()
                        state.heartbeat_generation = state.control_generation
                        state.online = True
                        restored = self.active_sessions_by_pc_id.get(assigned)
                        if restored:
                            state.current_user = str(restored.get("full_name", ""))
                            state.student_number = str(restored.get("student_number", ""))
                            state.year_section = str(restored.get("year_section", ""))
                            state.auth_session_id = int(restored.get("session_id", 0)) or None
                            state.session_timer = self._session_timer_from_payload(assigned, restored.get("session_timer"))
                            if state.session_timer is None:
                                deadline_ts = restored.get("deadline_ts")
                                if isinstance(deadline_ts, (int, float)):
                                    remaining_ms = max(0, int((float(deadline_ts) - time.time()) * 1000))
                                    state.session_timer = self._new_session_timer(assigned, remaining_ms)
                            state.interrupted = False
                            state.interrupted_remaining_s = 0
                            state.interrupted_until_ts = None
                            state.interrupted_student_number = ""
                    pc_id = assigned
                    self._send_control_json(assigned, client_sock, {
                        "type": "register_ack",
                        "pc_id": assigned,
                        "server_ts": time.time(),
                        "timers": self._active_timers(assigned),
                        "enable_session_messaging": bool(self.settings.enable_session_messaging),
                        "enable_extension_requests": bool(self.settings.enable_extension_requests),
                        "enable_timer_near_limit_notify": bool(self.settings.enable_timer_near_limit_notify),
                        "enable_timer_pause_on_temp_lock": bool(self.settings.enable_timer_pause_on_temp_lock),
                    })
                    self.send_command(assigned, "SET_STREAM_PROFILE", {
                        "width": {"360p": 640, "720p": 1280, "1080p": 1920}.get(self.settings.main_stream_profile, 1280),
                        "height": {"360p": 360, "720p": 720, "1080p": 1080}.get(self.settings.main_stream_profile, 720),
                        "jpeg_quality": {"360p": 60, "720p": 80, "1080p": 90}.get(self.settings.main_stream_profile, 80),
                        "max_fps": RUNTIME.max_fps,
                    })
                    self.send_command(assigned, "SET_RUNTIME_TUNING", {"reconnect_interval_s": int(self.settings.reconnect_interval_s)})
                    with self.lock:
                        active_share_id = self.screen_share_session_id if assigned in self.screen_share_targets else None
                    if active_share_id:
                        self.send_command(assigned, "SCREEN_SHARE_START", {
                            "session_id": active_share_id,
                            "sample_rate": RUNTIME.screen_share_audio_sample_rate,
                            "channels": RUNTIME.screen_share_audio_channels,
                        }, allow_udp_fallback=False)
                        self._log_event("SCREEN_SHARE_RECONNECT", pc_id=assigned, session_id=active_share_id)
                    if assigned in self.pending_signout_lock_pc_ids:
                        with self.lock:
                            client_ref = self.clients.get(assigned)
                            if client_ref:
                                self._apply_lock_reason(client_ref, "signout")
                        self.send_command(assigned, "LOCK_NOW", {"lock_mode": "signout", "pause_ts": time.time()})
                    self._put_bounded(self.status_queue, assigned)
                    self._log_event("client_registered", pc_id=assigned, hostname=hostname, mac=mac)
                elif msg_type == "heartbeat":
                    heart_id = str(msg.get("pc_id", ""))
                    if not heart_id:
                        self._record_error("CONTROL_VALIDATION_ERROR", "", "heartbeat missing pc_id")
                        continue
                    cpu_percent = msg.get("cpu_percent")
                    ram_percent = msg.get("ram_percent")
                    disk_percent = msg.get("disk_percent")
                    uptime_s = msg.get("uptime_s")
                    with self.lock:
                        if heart_id in self.clients:
                            client = self.clients[heart_id]
                            client.heartbeat.touch()
                            client.heartbeat_generation = client.control_generation
                            client.online = True
                            client.cpu_percent = float(cpu_percent) if isinstance(cpu_percent, (int, float)) else None
                            client.ram_percent = float(ram_percent) if isinstance(ram_percent, (int, float)) else None
                            client.disk_percent = float(disk_percent) if isinstance(disk_percent, (int, float)) else None
                            client.uptime_s = int(uptime_s) if isinstance(uptime_s, (int, float)) else None
                        else:
                            self._record_error("CONTROL_VALIDATION_ERROR", heart_id, "heartbeat for unknown pc_id")
                            continue
                    if heart_id:
                        self._put_bounded(self.status_queue, heart_id)
                elif msg_type == "ack":
                    self._handle_ack_message(msg)
                elif msg_type == "student_register" and pc_id:
                    self._handle_student_register(msg, client_sock, pc_id)
                elif msg_type == "student_login" and pc_id:
                    self._handle_student_login(msg, client_sock, pc_id)
                elif msg_type == "student_session_resume" and pc_id:
                    self._handle_session_resume(msg, client_sock, pc_id)
                elif msg_type == "student_session_message" and pc_id:
                    self._handle_student_session_message(msg, pc_id)
                elif msg_type == "student_extension_request" and pc_id:
                    self._handle_student_extension_request(msg, pc_id)
                elif msg_type == "extension_offer_response" and pc_id:
                    self._handle_extension_offer_response(msg, pc_id)
        except (ConnectionResetError, TimeoutError, OSError, Exception) as exc:
            self._log_event("control_client_error", pc_id=pc_id or "", reason=str(exc))
        finally:
            if file_obj is not None:
                try:
                    file_obj.close()
                except OSError:
                    pass
            if pc_id:
                with self.lock:
                    if pc_id in self.clients:
                        client = self.clients[pc_id]
                        if client.control_generation != owned_generation or client.control_sock is not client_sock:
                            client = None
                        if client is not None:
                            client.control_sock = None
                            client.online = False
                        if client is not None:
                            now_ts = time.time()
                            if client.current_user:
                                remaining_ms = self._timer_remaining_ms(client.session_timer, now_ts) if client.session_timer else 0
                                client.interrupted = True
                                client.interrupted_remaining_s = max(0, remaining_ms // 1000)
                                client.interrupted_until_ts = now_ts + RUNTIME.session_disconnect_grace_s
                                client.interrupted_student_number = client.student_number
                            else:
                                client.interrupted = False
                                client.interrupted_remaining_s = 0
                                client.interrupted_until_ts = None
                                client.interrupted_student_number = ""
                                client.session_timer = None
                if self.settings.recording_mode == "auto":
                    self._stop_recording(pc_id, status="disconnected")
                self._put_bounded(self.status_queue, pc_id)
                self._log_event("control_disconnected", pc_id=pc_id)
            client_sock.close()

    def _valid_student_number(self, value: str) -> bool:
        parts = value.strip().split("-")
        if len(parts) != 2:
            return False
        return parts[0].isdigit() and len(parts[0]) == 2 and parts[1].isdigit() and len(parts[1]) == 5

    def _send_control_json(self, pc_id: str, client_sock: socket.socket, payload: dict) -> bool:
        with self.lock:
            client = self.clients.get(pc_id)
            if not client or client.control_sock is not client_sock:
                return False
            send_lock = client.control_send_lock
        with send_lock:
            send_json(client_sock, payload)
        return True

    def _handle_student_register(self, msg: dict, client_sock: socket.socket, pc_id: str) -> None:
        full_name = str(msg.get("full_name", "")).strip()
        year_section = str(msg.get("year_section", "")).strip()
        student_number = str(msg.get("student_number", "")).strip()
        password = str(msg.get("password", ""))
        if not full_name or not year_section or not password or not self._valid_student_number(student_number):
            self._send_control_json(pc_id, client_sock, {"type": "auth_ack", "action": "register", "ok": False, "reason": "invalid_registration_fields"})
            return
        ok, reason = self.auth_db.register_student(full_name, year_section, student_number, password)
        self._send_control_json(pc_id, client_sock, {"type": "auth_ack", "action": "register", "ok": ok, "reason": reason})

    def _handle_student_login(self, msg: dict, client_sock: socket.socket, pc_id: str) -> None:
        student_number = str(msg.get("student_number", "")).strip()
        password = str(msg.get("password", ""))
        user = self.auth_db.verify_login(student_number, password)
        if not user:
            self._send_control_json(pc_id, client_sock, {"type": "auth_ack", "action": "login", "ok": False, "reason": "invalid_credentials"})
            return
        active_reservation = self.reservation_manager.get_active_reservation(pc_id, time.time())
        if active_reservation and str(active_reservation.get("student_number", "")).strip() != student_number:
            self._send_control_json(pc_id, client_sock, {
                "type": "auth_ack",
                "action": "login",
                "ok": False,
                "reason": "reserved_for_another_student",
                "message": "This workstation is reserved for another student during this time.",
            })
            return
        used_today_s = self.auth_db.get_today_usage_seconds(student_number)
        remaining_today_s = max(0, int(self.settings.daily_limit_s) - used_today_s)
        if remaining_today_s <= 0:
            self._send_control_json(pc_id, client_sock, {"type": "auth_ack", "action": "login", "ok": False, "reason": "daily_limit_reached"})
            return
        session_limit_s = min(int(self.settings.session_duration_s), remaining_today_s)
        session_duration_ms = int(session_limit_s) * 1000
        server_ts = time.time()
        with self.lock:
            client = self.clients.get(pc_id)
            existing_timer = client.session_timer if client else None
            existing_student_number = client.student_number if client else ""
        if client is None:
            send_json(client_sock, {"type": "auth_ack", "action": "login", "ok": False, "reason": "unknown_pc"})
            return

        session_id = self.auth_db.open_session(pc_id, user)

        login_aborted = False
        with self.lock:
            client = self.clients.get(pc_id)
            if client is None or client.control_sock is not client_sock:
                login_aborted = True
            else:
                if existing_timer is not None and existing_student_number == user["student_number"]:
                    existing_remaining_ms = self._timer_remaining_ms(existing_timer, server_ts)
                    if existing_remaining_ms > 0:
                        session_duration_ms = min(session_duration_ms, int(existing_remaining_ms))
                client.current_user = user["full_name"]
                client.year_section = user["year_section"]
                client.student_number = user["student_number"]
                client.auth_session_id = session_id
                client.session_timer = self._new_session_timer(pc_id, session_duration_ms, start_ts=server_ts)
                self._apply_lock_reason(client, None, clear=True)
                self._clear_signout_lock_after_auth(client, pc_id)
                client.interrupted = False
                client.interrupted_remaining_s = 0
                client.interrupted_until_ts = None
                client.interrupted_student_number = ""
        if login_aborted:
            self.auth_db.close_active_session(pc_id, status="login_aborted")
            send_json(client_sock, {"type": "auth_ack", "action": "login", "ok": False, "reason": "unknown_pc"})
            return
        self._send_control_json(pc_id, client_sock, {
            "type": "auth_ack",
            "action": "login",
            "ok": True,
            "user": user,
            "max_session_s": int(session_duration_ms // 1000),
            "server_ts": server_ts,
        })
        self._start_recording_for_session(pc_id, session_id)
        with self.lock:
            client = self.clients.get(pc_id)
            session_timer_payload = self._session_timer_payload(client.session_timer) if client else None
            login_ts = float(client.session_timer.start_ts) if (client and client.session_timer) else time.time()
            self.active_sessions_by_pc_id[pc_id] = {
                "session_id": session_id,
                "full_name": user["full_name"],
                "student_number": user["student_number"],
                "year_section": user["year_section"],
                "login_ts": login_ts,
                "session_timer": session_timer_payload,
            }
        self._put_bounded(self.status_queue, pc_id)
        self._persist_session_timers()

    def _handle_session_resume(self, msg: dict, client_sock: socket.socket, pc_id: str) -> None:
        user = msg.get("user", {}) if isinstance(msg.get("user"), dict) else {}
        full_name = str(user.get("full_name", "")).strip()
        year_section = str(user.get("year_section", "")).strip()
        student_number = str(user.get("student_number", "")).strip()
        if not full_name or not year_section or not self._valid_student_number(student_number):
            self._send_control_json(pc_id, client_sock, {"type": "auth_ack", "action": "resume", "ok": False, "reason": "invalid_resume_payload"})
            return
        active_reservation = self.reservation_manager.get_active_reservation(pc_id, time.time())
        if active_reservation and str(active_reservation.get("student_number", "")).strip() != student_number:
            self._send_control_json(pc_id, client_sock, {
                "type": "auth_ack",
                "action": "resume",
                "ok": False,
                "reason": "reserved_for_another_student",
                "message": "This workstation is reserved for another student during this time.",
            })
            return
        with self.lock:
            client = self.clients.get(pc_id)
            if not client:
                send_json(client_sock, {"type": "auth_ack", "action": "resume", "ok": False, "reason": "unknown_pc"})
                return
            if not client.current_user or not client.student_number:
                restored = self.active_sessions_by_pc_id.get(pc_id)
                if restored:
                    client.current_user = str(restored.get("full_name", ""))
                    client.year_section = str(restored.get("year_section", ""))
                    client.student_number = str(restored.get("student_number", ""))
                    client.auth_session_id = int(restored.get("session_id", 0)) or None
                    client.session_timer = self._session_timer_from_payload(pc_id, restored.get("session_timer"))
                    if client.session_timer is None:
                        deadline_ts = restored.get("deadline_ts")
                        if isinstance(deadline_ts, (int, float)):
                            remaining_ms = max(0, int((float(deadline_ts) - time.time()) * 1000))
                            client.session_timer = self._new_session_timer(pc_id, remaining_ms)
                    client.interrupted = False
                    client.interrupted_remaining_s = 0
                    client.interrupted_until_ts = None
                    client.interrupted_student_number = ""
                else:
                    with client.control_send_lock:
                        send_json(client_sock, {"type": "auth_ack", "action": "resume", "ok": False, "reason": "no_resumable_session"})
                    return
            if client.interrupted and client.interrupted_student_number and client.interrupted_student_number != student_number:
                with client.control_send_lock:
                    send_json(client_sock, {"type": "auth_ack", "action": "resume", "ok": False, "reason": "resume_mismatch"})
                return
            if (
                client.current_user != full_name
                or client.year_section != year_section
                or client.student_number != student_number
            ):
                with client.control_send_lock:
                    send_json(client_sock, {"type": "auth_ack", "action": "resume", "ok": False, "reason": "resume_mismatch"})
                return
            session_timer = client.session_timer
            if (
                (not client.interrupted)
                and client.auth_session_id
                and session_timer is not None
                and self._timer_remaining_ms(session_timer, time.time()) > 0
            ):
                self.active_sessions_by_pc_id[pc_id] = {
                    "session_id": int(client.auth_session_id or 0),
                    "full_name": client.current_user,
                    "student_number": client.student_number,
                    "year_section": client.year_section,
                    "login_ts": float(session_timer.start_ts),
                    "session_timer": self._session_timer_payload(session_timer),
                }
                with client.control_send_lock:
                    send_json(client_sock, {"type": "auth_ack", "action": "resume", "ok": True, "server_ts": time.time()})
                self._put_bounded(self.status_queue, pc_id)
                self._persist_session_timers()
                return
            session_id = self.auth_db.open_session(pc_id, {
                "full_name": client.current_user,
                "year_section": client.year_section,
                "student_number": client.student_number,
            })
            client.auth_session_id = session_id
            existing_timer = client.session_timer
            was_paused = bool(existing_timer and existing_timer.paused)
            paused_at_ts = existing_timer.paused_at_ts if existing_timer else None
            paused_accum_ms = int(existing_timer.paused_accum_ms) if existing_timer else 0
            if client.interrupted and client.interrupted_remaining_s > 0:
                client.session_timer = self._new_session_timer(pc_id, int(client.interrupted_remaining_s) * 1000)
            elif existing_timer is None:
                client.session_timer = self._new_session_timer(pc_id, int(self.settings.session_duration_s) * 1000)
            if client.session_timer and was_paused:
                client.session_timer.paused = True
                client.session_timer.paused_at_ts = paused_at_ts
                client.session_timer.paused_accum_ms = paused_accum_ms
            self._apply_lock_reason(client, None, clear=True)
            self._clear_signout_lock_after_auth(client, pc_id)
            client.interrupted = False
            client.interrupted_remaining_s = 0
            client.interrupted_until_ts = None
            client.interrupted_student_number = ""
        self.active_sessions_by_pc_id[pc_id] = {
            "session_id": session_id,
            "full_name": client.current_user,
            "student_number": client.student_number,
            "year_section": client.year_section,
            "login_ts": float(client.session_timer.start_ts) if client.session_timer else time.time(),
            "session_timer": self._session_timer_payload(client.session_timer),
        }
        self._send_control_json(pc_id, client_sock, {"type": "auth_ack", "action": "resume", "ok": True, "server_ts": time.time()})
        self._start_recording_for_session(pc_id, session_id)
        self._put_bounded(self.status_queue, pc_id)
        self._persist_session_timers()

    def _timer_remaining_ms(self, timer: TimerState, now: Optional[float] = None) -> int:
        ts_now = time.time() if now is None else float(now)
        elapsed_ms = int((ts_now - timer.start_ts) * 1000)
        paused_ms = int(timer.paused_accum_ms)
        if timer.paused and timer.paused_at_ts is not None:
            paused_ms += max(0, int((ts_now - timer.paused_at_ts) * 1000))
        effective_elapsed = max(0, elapsed_ms - paused_ms)
        return max(0, int(timer.duration_ms) - effective_elapsed)

    def _pause_timers_for_targets(self, targets: list[str]) -> None:
        now = time.time()
        target_set = set(targets)
        changed = False
        with self.lock:
            for timer in self.timers.values():
                if not timer.active:
                    continue
                if not any(pc in target_set for pc in timer.targets):
                    continue
                if timer.paused:
                    continue
                timer.paused = True
                timer.paused_at_ts = now
                changed = True
        if changed:
            for pc_id in targets:
                for timer in list(self.timers.values()):
                    if timer.active and timer.paused and pc_id in timer.targets:
                        self.send_command(pc_id, "PAUSE_TIMER", {"timer_id": timer.timer_id, "pause_ts": now})
            self._persist_timers()

    def _resume_timers_for_targets(self, targets: list[str]) -> None:
        now = time.time()
        target_set = set(targets)
        changed = False
        with self.lock:
            for timer in self.timers.values():
                if not timer.active:
                    continue
                if not any(pc in target_set for pc in timer.targets):
                    continue
                if not timer.paused:
                    continue
                if timer.paused_at_ts is not None:
                    timer.paused_accum_ms += max(0, int((now - timer.paused_at_ts) * 1000))
                timer.paused_at_ts = None
                timer.paused = False
                changed = True
        if changed:
            for pc_id in targets:
                for timer in list(self.timers.values()):
                    if timer.active and pc_id in timer.targets:
                        self.send_command(pc_id, "RESUME_TIMER", {"timer_id": timer.timer_id, "resume_ts": now})
            self._persist_timers()

    def _handle_ack_message(self, msg: dict) -> None:
        ack_id = str(msg.get("pc_id", ""))
        command = str(msg.get("command", ""))
        cmd_id = str(msg.get("cmd_id", "")).strip()
        result = str(msg.get("result", "applied")).strip().lower() or "applied"
        reason = str(msg.get("reason", "")).strip()
        if not ack_id or command not in VALID_COMMAND_ACKS:
            self._record_error("CONTROL_VALIDATION_ERROR", ack_id, f"ack validation failed command={command}")
            return
        if result not in {"applied", "failed"}:
            self._record_error("CONTROL_VALIDATION_ERROR", ack_id, f"ack invalid result={result}")
            return
        with self.lock:
            if ack_id in self.clients:
                client = self.clients[ack_id]
                client.last_ack = command
                client.last_ack_result = result
                if command == "TIMER_EXPIRED" and result == "applied":
                    changed_timer = False
                    now_ts = time.time()
                    for timer in self.timers.values():
                        if timer.active and ack_id in timer.targets:
                            remaining_ms = self._timer_remaining_ms(timer, now_ts)
                            if remaining_ms <= 0:
                                timer.active = False
                                changed_timer = True
                    if changed_timer:
                        if client.lock_reason != "signout":
                            self._apply_lock_reason(client, "timer")
                        self._persist_timers()
            else:
                self._record_error("CONTROL_VALIDATION_ERROR", ack_id, "ack for unknown pc_id")
                return
        self._log_event("command_ack", pc_id=ack_id, command=command, cmd_id=cmd_id, result=result, reason=reason)
        self._put_bounded(self.status_queue, ack_id)

    def _clear_conversation_for_pc(self, pc_id: str) -> None:
        self.conversations.pop(pc_id, None)

    def _prune_conversation_retention(self, now: Optional[float] = None) -> None:
        ts_now = time.time() if now is None else float(now)
        stale_before = ts_now - CONVERSATION_TTL_S
        # Remove stale conversations for inactive/offline PCs.
        removable = []
        for pc_id, convo in self.conversations.items():
            last_ts = float(convo.get("last_ts", 0.0) or 0.0)
            client = self.clients.get(pc_id)
            is_active = bool(client and (client.current_user or client.online or client.interrupted))
            if (not is_active) and last_ts < stale_before:
                removable.append(pc_id)
        for pc_id in removable:
            self.conversations.pop(pc_id, None)

        if len(self.conversations) > CONVERSATION_MAX_PCS:
            ordered = sorted(self.conversations.items(), key=lambda item: float(item[1].get("last_ts", 0.0) or 0.0))
            overflow = len(self.conversations) - CONVERSATION_MAX_PCS
            for pc_id, _ in ordered[:overflow]:
                self.conversations.pop(pc_id, None)

    def _append_conversation_message(self, pc_id: str, session_id: Optional[int], direction: str, text: str, ts: Optional[float] = None) -> None:
        now = time.time() if ts is None else float(ts)
        convo = self.conversations.get(pc_id)
        if convo is None:
            convo = {"session_id": session_id, "messages": [], "unread": 0, "last_ts": now}
            self.conversations[pc_id] = convo
        if session_id is not None:
            convo["session_id"] = session_id
        convo["last_ts"] = now
        messages = convo.setdefault("messages", [])
        messages.append({"ts": now, "direction": direction, "text": text, "session_id": session_id})
        if len(messages) > 500:
            del messages[:-500]
        if direction == "student_to_teacher":
            convo["unread"] = int(convo.get("unread", 0)) + 1

    def get_conversation_snapshot(self, pc_id: str, mark_read: bool = False) -> dict:
        with self.lock:
            convo = self.conversations.get(pc_id)
            if not convo:
                return {"pc_id": pc_id, "session_id": None, "unread": 0, "messages": []}
            if mark_read:
                convo["unread"] = 0
            return {
                "pc_id": pc_id,
                "session_id": convo.get("session_id"),
                "unread": int(convo.get("unread", 0)),
                "last_ts": float(convo.get("last_ts", 0.0) or 0.0),
                "messages": [dict(m) for m in list(convo.get("messages", []))],
            }

    def _handle_student_session_message(self, msg: dict, pc_id: str) -> None:
        if not self.settings.enable_session_messaging:
            return
        text = str(msg.get("text", "")).strip()
        if not text:
            return
        with self.lock:
            client = self.clients.get(pc_id)
            if not client or not client.current_user:
                return
            event = {
                "event_id": f"evt-{uuid.uuid4().hex[:10]}",
                "ts": time.time(),
                "pc_id": pc_id,
                "session_id": client.auth_session_id,
                "direction": "student_to_teacher",
                "text": text,
                "status": "delivered",
            }
            self.message_events.append(event)
            self._append_conversation_message(pc_id, client.auth_session_id, "student_to_teacher", text, event["ts"])
        self._put_bounded(self.status_queue, pc_id)
        self._log_event("CHAT_UNREAD_SET", pc_id=pc_id)
        self._log_event("student_session_message", pc_id=pc_id, text=text)

    def _handle_student_extension_request(self, msg: dict, pc_id: str) -> None:
        if not self.settings.enable_extension_requests:
            return
        try:
            requested_extra_ms = int(msg.get("requested_extra_ms", 0))
        except (TypeError, ValueError):
            requested_extra_ms = 0
        requested_extra_ms = max(60_000, requested_extra_ms)
        with self.lock:
            client = self.clients.get(pc_id)
            if not client or not client.current_user or not client.auth_session_id:
                return
            req_id = f"ext-{uuid.uuid4().hex[:10]}"
            req = {
                "request_id": req_id,
                "ts": time.time(),
                "pc_id": pc_id,
                "session_id": client.auth_session_id,
                "student_number": client.student_number,
                "student_name": client.current_user,
                "requested_extra_ms": requested_extra_ms,
                "status": "requested",
            }
            self.extension_requests[req_id] = req
            self.extension_status_by_pc[pc_id] = "requested"
            self.message_events.append({
                "event_id": f"evt-{uuid.uuid4().hex[:10]}",
                "ts": time.time(),
                "pc_id": pc_id,
                "session_id": client.auth_session_id,
                "direction": "student_to_teacher",
                "text": f"Extension requested: +{requested_extra_ms // 60000} minute(s)",
                "status": "delivered",
            })
        self._put_bounded(self.status_queue, pc_id)
        self._log_event("student_extension_request", pc_id=pc_id, requested_extra_ms=requested_extra_ms)

    def send_session_message(self, pc_id: str, text: str) -> Optional[str]:
        if not self.settings.enable_session_messaging:
            return None
        text = text.strip()
        if not text:
            return None
        with self.lock:
            client = self.clients.get(pc_id)
            if not client or not client.current_user:
                return None
        cmd_id = self.send_command(pc_id, "SESSION_MESSAGE", {"text": text})
        status = "sent" if cmd_id else "failed"
        with self.lock:
            session_id = self.clients.get(pc_id).auth_session_id if pc_id in self.clients else None
            event_ts = time.time()
            self.message_events.append({
                "event_id": f"evt-{uuid.uuid4().hex[:10]}",
                "ts": event_ts,
                "pc_id": pc_id,
                "session_id": session_id,
                "direction": "teacher_to_student",
                "text": text,
                "status": status,
                "cmd_id": cmd_id or "",
            })
            self._append_conversation_message(pc_id, session_id, "teacher_to_student", text, event_ts)
        self._put_bounded(self.status_queue, pc_id)
        return cmd_id

    def respond_extension_request(self, request_id: str, approved: bool, extra_ms: Optional[int] = None) -> bool:
        if not self.settings.enable_extension_requests:
            return False
        with self.lock:
            req = self.extension_requests.get(request_id)
            if not req:
                return False
            if req.get("status") not in {"requested", "teacher_sent"}:
                return False
            pc_id = str(req.get("pc_id", ""))
            effective_extra_ms = int(extra_ms if extra_ms is not None else req.get("requested_extra_ms", 0))
            effective_extra_ms = max(60_000, effective_extra_ms)
            req["status"] = "teacher_sent"
        payload = {
            "request_id": request_id,
            "approved": bool(approved),
            "extra_ms": effective_extra_ms,
        }
        cmd_id = self.send_command(pc_id, "EXTENSION_OFFER", payload)
        with self.lock:
            if cmd_id:
                self.extension_status_by_pc[pc_id] = "teacher_sent"
                req = self.extension_requests.get(request_id)
                if req:
                    req["cmd_id"] = cmd_id
            else:
                self.extension_status_by_pc[pc_id] = "failed"
                req = self.extension_requests.get(request_id)
                if req:
                    req["status"] = "failed"
        self._put_bounded(self.status_queue, pc_id)
        return bool(cmd_id)

    def _handle_extension_offer_response(self, msg: dict, pc_id: str) -> None:
        if not self.settings.enable_extension_requests:
            return
        request_id = str(msg.get("request_id", "")).strip()
        decision = str(msg.get("decision", "")).strip().lower()
        if not request_id or decision not in {"accepted", "declined"}:
            return
        with self.lock:
            req = self.extension_requests.get(request_id)
            if not req or req.get("pc_id") != pc_id:
                return
            if decision == "accepted":
                req["status"] = "student_accepted"
                self.extension_status_by_pc[pc_id] = "student_accepted"
                extra_ms = int(req.get("requested_extra_ms", 60_000))
            else:
                req["status"] = "student_declined"
                self.extension_status_by_pc[pc_id] = "student_declined"
                extra_ms = 0
        if decision == "accepted":
            self.extend_timer([pc_id], extra_ms)
            self._log_event("extension_request_applied", pc_id=pc_id, request_id=request_id, extra_ms=extra_ms)
            with self.lock:
                req = self.extension_requests.get(request_id)
                if req:
                    req["status"] = "applied"
                self.extension_status_by_pc[pc_id] = "applied"
        else:
            self._log_event("extension_request_declined", pc_id=pc_id, request_id=request_id)
        self._put_bounded(self.status_queue, pc_id)

    def _run_udp_fallback_server(self) -> None:
        while True:
            sock: Optional[socket.socket] = None
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.bind((self.settings.teacher_bind_host, NETWORK.command_fallback_port))
                while True:
                    data, _ = sock.recvfrom(4096)
                    try:
                        msg = json.loads(data.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue
                    if msg.get("type") == "ack":
                        self._handle_ack_message(msg)
            except OSError as exc:
                self._log_event("udp_fallback_restart", reason=str(exc))
                time.sleep(1)
            finally:
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass

    def _run_video_server(self) -> None:
        while True:
            sock: Optional[socket.socket] = None
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((self.settings.teacher_bind_host, NETWORK.video_port))
                sock.listen()
                while True:
                    client_sock, _ = sock.accept()
                    threading.Thread(target=self._video_client_loop, args=(client_sock,), daemon=True).start()
            except OSError as exc:
                self._log_event("video_server_restart", reason=str(exc))
                time.sleep(1)
            finally:
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass

    def _send_udp_fallback(self, msg: dict, peer_ip: str) -> bool:
        payload = json.dumps(msg).encode("utf-8")
        with self.udp_send_lock:
            if self.udp_send_sock is None:
                try:
                    self.udp_send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                except OSError:
                    self.udp_send_sock = None
                    return False
            try:
                self.udp_send_sock.sendto(payload, (peer_ip, NETWORK.command_fallback_port))
                return True
            except OSError:
                try:
                    if self.udp_send_sock is not None:
                        self.udp_send_sock.close()
                except OSError:
                    pass
                self.udp_send_sock = None
                return False

    def _video_client_loop(self, client_sock: socket.socket) -> None:
        pc_id: Optional[str] = None
        file_obj = None
        role = "uplink"
        try:
            file_obj = client_sock.makefile("rb")
            reg = recv_json_line(file_obj)
            if not reg or reg.get("type") != "video_register":
                return
            pc_id = str(reg.get("pc_id", ""))
            role = str(reg.get("role", "uplink")).strip().lower() or "uplink"
            if role != "uplink":
                return
            registration_diag: dict[str, object] = {"pc_id": pc_id, "role": role, "accepted_socket_id": id(client_sock)}
            with self.lock:
                if pc_id not in self.clients:
                    self._log_event("video_registration_unknown_client", pc_id=pc_id)
                    return
                client = self.clients[pc_id]
                if client.video_sock and client.video_sock is not client_sock:
                    try:
                        client.video_sock.close()
                    except OSError:
                        pass
                client.video_generation += 1
                owned_video_generation = client.video_generation
                client.video_sock = client_sock
            self._log_event("video_registered", pc_id=pc_id)
            while True:
                with self.lock:
                    client = self.clients.get(pc_id or "")
                    if not client or client.video_generation != owned_video_generation or client.video_sock is not client_sock:
                        break
                frame_data = recv_frame(client_sock)
                if frame_data is None:
                    break
                image = self._decode_frame(frame_data)
                if image is None:
                    if pc_id:
                        self._record_error("VIDEO_DECODE_ERROR", pc_id, "failed to decode incoming frame")
                    continue
                with self.lock:
                    if pc_id in self.clients:
                        self.clients[pc_id].last_frame = image
                        self.clients[pc_id].online = True
                    rec = self.recordings.get(pc_id)
                if rec is not None:
                    arr = numpy.array(image)
                    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    if (bgr.shape[1], bgr.shape[0]) != (RUNTIME.frame_width, RUNTIME.frame_height):
                        bgr = cv2.resize(bgr, (RUNTIME.frame_width, RUNTIME.frame_height))
                    rec.writer.write(bgr)
                self._put_bounded(self.frame_queue, (pc_id, image))
                now_ts = time.time()
                should_emit_status = False
                with self.lock:
                    last_ts = float(self.video_status_last_ts_by_pc.get(pc_id, 0.0))
                    if (now_ts - last_ts) >= 1.0:
                        self.video_status_last_ts_by_pc[pc_id] = now_ts
                        should_emit_status = True
                if should_emit_status:
                    self._put_bounded(self.status_queue, pc_id)
        except (ConnectionResetError, TimeoutError, OSError, Exception) as exc:
            self._log_event("video_client_error", pc_id=pc_id or "", reason=str(exc))
        finally:
            if file_obj is not None:
                try:
                    file_obj.close()
                except OSError:
                    pass
            if pc_id:
                with self.lock:
                    if pc_id in self.clients and self.clients[pc_id].video_sock is client_sock:
                        self.clients[pc_id].video_sock = None
                    self.video_status_last_ts_by_pc.pop(pc_id, None)
                self._put_bounded(self.status_queue, pc_id)
                self._log_event("video_disconnected", pc_id=pc_id)
            client_sock.close()

    def _run_sensor_server(self) -> None:
        while True:
            sock: Optional[socket.socket] = None
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.bind((self.settings.teacher_bind_host, NETWORK.sensor_port))
                while True:
                    data, _ = sock.recvfrom(4096)
                    try:
                        payload = json.loads(data.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue
                    pc_id = str(payload.get("pc_id", ""))
                    if not pc_id:
                        self._record_error("SENSOR_VALIDATION_ERROR", "", "sensor payload missing pc_id")
                        continue
                    temperature = payload.get("temperature")
                    fan_ok = payload.get("fan_ok")
                    fan_rpm = payload.get("fan_rpm")
                    if temperature is not None and not isinstance(temperature, (int, float)):
                        self._record_error("SENSOR_VALIDATION_ERROR", pc_id, "temperature is not numeric")
                        continue
                    if fan_ok is not None and not isinstance(fan_ok, bool):
                        self._record_error("SENSOR_VALIDATION_ERROR", pc_id, "fan_ok is not boolean")
                        continue
                    if fan_rpm is not None and not isinstance(fan_rpm, int):
                        self._record_error("SENSOR_VALIDATION_ERROR", pc_id, "fan_rpm is not integer")
                        continue
                    if isinstance(temperature, (int, float)) and (temperature < -20 or temperature > 120):
                        self._record_error("SENSOR_OUTLIER", pc_id, f"temperature outlier={temperature}")
                        continue
                    if isinstance(fan_rpm, int) and (fan_rpm < 0 or fan_rpm > 20000):
                        self._record_error("SENSOR_OUTLIER", pc_id, f"fan_rpm outlier={fan_rpm}")
                        continue
                    is_unknown_sensor = False
                    with self.lock:
                        if pc_id not in self.sensors:
                            is_unknown_sensor = True
                        else:
                            sensor = self.sensors[pc_id]
                            sensor.temperature = float(temperature) if isinstance(temperature, (int, float)) else None
                            sensor.fan_ok = fan_ok if isinstance(fan_ok, bool) else None
                            sensor.fan_rpm = fan_rpm if isinstance(fan_rpm, int) else None
                            sensor.last_update = time.time()
                    if is_unknown_sensor:
                        self._record_error("SENSOR_VALIDATION_ERROR", pc_id, "sensor payload for unknown pc_id")
                        continue
                    self._put_bounded(self.sensor_queue, pc_id)
            except OSError as exc:
                self._log_event("sensor_server_restart", reason=str(exc))
                time.sleep(1)
            finally:
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass

    def _monitor_heartbeats(self) -> None:
        while True:
            with self.lock:
                ids = list(self.clients.keys())
                for pc_id in ids:
                    client = self.clients[pc_id]
                    online = client.heartbeat.is_online(int(self.settings.heartbeat_timeout_s))
                    if not online:
                        if client.control_sock is not None and client.heartbeat_generation != client.control_generation:
                            continue
                        client.online = False
                        client.last_frame = None
                        self._put_bounded(self.status_queue, pc_id)
            time.sleep(1)

    def _monitor_student_sessions(self) -> None:
        next_retention_prune_ts = time.time() + 60.0
        while True:
            now = time.time()
            expired: list[str] = []
            interrupted_expired: list[str] = []
            with self.lock:
                for pc_id, client in self.clients.items():
                    if client.interrupted and client.interrupted_until_ts:
                        client.interrupted_remaining_s = max(0, int(client.interrupted_until_ts - now))
                    session_timer = client.session_timer
                    remaining_s = (self._timer_remaining_ms(session_timer, now) // 1000) if session_timer is not None else None
                    if session_timer is not None and (not session_timer.paused) and remaining_s is not None and client.current_user and remaining_s <= 0:
                        expired.append(pc_id)
                    elif (
                        client.interrupted
                        and client.interrupted_until_ts
                        and client.current_user
                        and now >= client.interrupted_until_ts
                    ):
                        interrupted_expired.append(pc_id)
            for pc_id in expired:
                self.send_command(pc_id, "LOCK_NOW", {"lock_mode": "signout"})
                with self.lock:
                    client = self.clients.get(pc_id)
                    if not client:
                        continue
                    self.auth_db.close_active_session(pc_id, status="session_timeout")
                    client.current_user = ""
                    client.year_section = ""
                    client.student_number = ""
                    client.auth_session_id = None
                    client.session_timer = None
                    client.interrupted = False
                    client.interrupted_remaining_s = 0
                    client.interrupted_until_ts = None
                    client.interrupted_student_number = ""
                    self.active_sessions_by_pc_id.pop(pc_id, None)
                    self._clear_conversation_for_pc(pc_id)
                    self._apply_lock_reason(client, "signout")
                self._stop_recording(pc_id, status="session_timeout")
                self._put_bounded(self.status_queue, pc_id)
            for pc_id in interrupted_expired:
                with self.lock:
                    client = self.clients.get(pc_id)
                    if not client or not client.current_user:
                        continue
                    self.auth_db.close_active_session(pc_id, status="disconnected")
                    client.current_user = ""
                    client.year_section = ""
                    client.student_number = ""
                    client.auth_session_id = None
                    client.session_timer = None
                    client.interrupted = False
                    client.interrupted_remaining_s = 0
                    client.interrupted_until_ts = None
                    client.interrupted_student_number = ""
                    self.active_sessions_by_pc_id.pop(pc_id, None)
                    self._clear_conversation_for_pc(pc_id)
                self._stop_recording(pc_id, status="disconnected")
                self._put_bounded(self.status_queue, pc_id)
            if expired or interrupted_expired:
                self._persist_session_timers()
            if now >= next_retention_prune_ts:
                with self.lock:
                    self._prune_conversation_retention(now)
                next_retention_prune_ts = now + 60.0
            time.sleep(1)

    def get_sessions_for_pc(self, pc_id: str, limit: int = 200) -> list[dict]:
        return self.auth_db.get_sessions_for_pc(pc_id, limit=limit)

    def get_all_sessions(self, limit: int = 500) -> list[dict]:
        return self.auth_db.get_all_sessions(limit=limit)

    def update_settings(self, settings: AppSettings) -> None:
        self.settings = settings
        self.default_session_s = int(settings.session_duration_s)
        self.settings_store.save(settings)
        self._apply_stream_profile_to_all()
        with self.lock:
            ids = list(self.clients.keys())
        for pc_id in ids:
            self.send_command(pc_id, "SET_RUNTIME_TUNING", {
                "reconnect_interval_s": int(settings.reconnect_interval_s),
            })
        self._apply_recording_retention()

    def _decode_frame(self, payload: bytes) -> Optional[Image.Image]:
        np_buf = numpy.frombuffer(payload, dtype=numpy.uint8)
        frame = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _recordings_dir(self) -> Path:
        path = ROOT / "recordings"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _apply_recording_retention(self) -> None:
        keep_seconds = max(1, int(self.settings.recording_retention_days)) * 86400
        max_bytes = int(max(0.1, float(self.settings.recording_max_gb)) * 1024 * 1024 * 1024)
        base = self._recordings_dir()
        files = [f for f in base.glob("*.mp4") if f.is_file()]
        now = time.time()
        for f in files:
            try:
                if now - f.stat().st_mtime > keep_seconds:
                    f.unlink(missing_ok=True)
            except OSError:
                pass
        files = sorted([f for f in base.glob("*.mp4") if f.is_file()], key=lambda x: x.stat().st_mtime)
        total = sum(f.stat().st_size for f in files)
        while total > max_bytes and files:
            old = files.pop(0)
            try:
                size = old.stat().st_size
                old.unlink(missing_ok=True)
                total -= size
            except OSError:
                break

    def _start_recording_for_session(self, pc_id: str, session_id: Optional[int], force: bool = False) -> None:
        if (not force) and self.settings.recording_mode != "auto":
            return
        if session_id is None:
            return
        with self.lock:
            if pc_id in self.recordings:
                return
        path = self._recordings_dir() / f"{pc_id}_{session_id}_{int(time.time())}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(1, RUNTIME.max_fps), (RUNTIME.frame_width, RUNTIME.frame_height))
        if not writer.isOpened():
            return
        rec_id = self.auth_db.open_recording(session_id, pc_id, str(path))
        with self.lock:
            self.recordings[pc_id] = RecordingState(rec_id, session_id, pc_id, str(path), writer, time.time())

    def _stop_recording(self, pc_id: str, status: str = "closed") -> None:
        with self.lock:
            rec = self.recordings.pop(pc_id, None)
        if not rec:
            return
        try:
            rec.writer.release()
        finally:
            self.auth_db.close_recording(rec.recording_id, status=status)
            self._apply_recording_retention()

    def _apply_stream_profile_to_all(self) -> None:
        profile_map = {"360p": (640, 360, 60), "720p": (1280, 720, 80), "1080p": (1920, 1080, 90)}
        w, h, q = profile_map.get(self.settings.main_stream_profile, profile_map["720p"])
        with self.lock:
            ids = list(self.clients.keys())
        for pc_id in ids:
            self.send_command(pc_id, "SET_STREAM_PROFILE", {
                "width": w,
                "height": h,
                "jpeg_quality": q,
                "max_fps": RUNTIME.max_fps,
            })

    def eligible_screen_share_targets(self) -> list[str]:
        with self.lock:
            return [pc_id for pc_id, client in self.clients.items() if client.online and client.control_sock is not None]

    def start_screen_share(self) -> bool:
        targets = self.eligible_screen_share_targets()
        if not targets:
            self._log_event("SCREEN_SHARE_START_REQUESTED", result="no_eligible_students")
            return False
        with self.lock:
            if self.screen_share_session_id:
                return True
            session_id, now = uuid.uuid4().hex, time.time()
            self.screen_share_session_id, self.screen_share_targets = session_id, set(targets)
            self.screen_share_timer_paused_by_us.clear(); self.screen_share_stop_event.clear()
            for pc_id in targets:
                client = self.clients.get(pc_id)
                if client and client.current_user and client.session_timer and not client.session_timer.paused:
                    client.session_timer.paused, client.session_timer.paused_at_ts = True, now
                    self.screen_share_timer_paused_by_us.add(pc_id)
        self._persist_session_timers()
        payload = {"session_id": session_id, "sample_rate": RUNTIME.screen_share_audio_sample_rate, "channels": RUNTIME.screen_share_audio_channels}
        for pc_id in targets:
            self.send_command(pc_id, "SCREEN_SHARE_START", payload, allow_udp_fallback=False)
        threading.Thread(target=self._screen_share_video_capture_loop, args=(session_id,), daemon=True).start()
        threading.Thread(target=self._screen_share_microphone_loop, args=(session_id,), daemon=True).start()
        self._log_event("SCREEN_SHARE_STARTED", session_id=session_id, targets=len(targets))
        return True

    def stop_screen_share(self) -> None:
        with self.lock:
            session_id = self.screen_share_session_id
            if not session_id: return
            targets, resume, now = list(self.screen_share_targets), list(self.screen_share_timer_paused_by_us), time.time()
            self.screen_share_session_id = None; self.screen_share_targets.clear(); self.screen_share_timer_paused_by_us.clear(); self.screen_share_stop_event.set()
            self.screen_share_video_queues.clear(); self.screen_share_audio_queues.clear()
            for pc_id in resume:
                client = self.clients.get(pc_id); timer = client.session_timer if client else None
                if timer and timer.paused:
                    if timer.paused_at_ts is not None: timer.paused_accum_ms += max(0, int((now - timer.paused_at_ts) * 1000))
                    timer.paused, timer.paused_at_ts = False, None
        self._persist_session_timers()
        for pc_id in targets: self.send_command(pc_id, "SCREEN_SHARE_STOP", {"session_id": session_id}, allow_udp_fallback=False)
        self._log_event("SCREEN_SHARE_STOPPED", session_id=session_id)

    def _screen_share_fanout(self, queues: dict[str, queue.Queue[bytes]], payload: bytes) -> None:
        with self.lock: recipients = list(queues.items())
        for _pc_id, out_q in recipients: self._put_bounded(out_q, payload)

    def _screen_share_video_capture_loop(self, session_id: str) -> None:
        try:
            with mss.mss() as capture:
                monitor = capture.monitors[1]
                while not self.screen_share_stop_event.is_set() and self.screen_share_session_id == session_id:
                    frame = cv2.cvtColor(numpy.array(capture.grab(monitor)), cv2.COLOR_BGRA2BGR)
                    ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    if ok: self._screen_share_fanout(self.screen_share_video_queues, jpeg.tobytes())
                    self.screen_share_stop_event.wait(1 / max(1, RUNTIME.max_fps))
        except Exception as exc: self._log_event("screen_share_video_capture_failed", session_id=session_id, reason=str(exc))

    def _screen_share_microphone_loop(self, session_id: str) -> None:
        # RawInputStream uses sounddevice's default input (microphone), never output/loopback audio.
        def callback(indata, _frames, _time_info, status) -> None:
            if status: self._log_event("microphone_capture_status", session_id=session_id, detail=str(status))
            if self.screen_share_session_id == session_id and not self.screen_share_stop_event.is_set(): self._screen_share_fanout(self.screen_share_audio_queues, bytes(indata))
        try:
            with sd.RawInputStream(samplerate=RUNTIME.screen_share_audio_sample_rate, channels=RUNTIME.screen_share_audio_channels, dtype="int16", blocksize=RUNTIME.screen_share_audio_blocksize, callback=callback):
                self._log_event("MICROPHONE_CAPTURE_STARTED", session_id=session_id)
                while not self.screen_share_stop_event.wait(.2) and self.screen_share_session_id == session_id: pass
        except Exception as exc: self._log_event("MICROPHONE_CAPTURE_FAILED", session_id=session_id, reason=str(exc))
        finally: self._log_event("MICROPHONE_CAPTURE_STOPPED", session_id=session_id)

    def _run_screen_share_media_server(self, port: int, media_type: str) -> None:
        while True:
            sock = None
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM); sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); sock.bind((self.settings.teacher_bind_host, port)); sock.listen()
                while True:
                    client_sock, _ = sock.accept(); threading.Thread(target=self._screen_share_media_client_loop, args=(client_sock, media_type), daemon=True).start()
            except OSError as exc: self._log_event("screen_share_media_server_restart", media=media_type, reason=str(exc)); time.sleep(1)
            finally:
                if sock: sock.close()

    def _screen_share_media_client_loop(self, client_sock: socket.socket, media_type: str) -> None:
        out_q = None; pc_id = ""
        try:
            file_obj = client_sock.makefile("rb"); registration = recv_json_line(file_obj)
            if not registration or registration.get("type") != "screen_share_register": return
            pc_id, session_id = str(registration.get("pc_id", "")), str(registration.get("session_id", ""))
            with self.lock:
                if not pc_id or session_id != self.screen_share_session_id or pc_id not in self.screen_share_targets: return
                queues = self.screen_share_video_queues if media_type == "video" else self.screen_share_audio_queues
                out_q = queue.Queue(maxsize=RUNTIME.screen_share_queue_max); queues[pc_id] = out_q
            self._log_event("STUDENT_SCREEN_SHARE_CONNECTED", pc_id=pc_id, media=media_type, session_id=session_id)
            while self.screen_share_session_id == session_id and not self.screen_share_stop_event.is_set():
                try: send_frame(client_sock, out_q.get(timeout=.5))
                except queue.Empty: continue
        except (OSError, ValueError) as exc: self._log_event("screen_share_media_client_error", pc_id=pc_id, media=media_type, reason=str(exc))
        finally:
            with self.lock:
                queues = self.screen_share_video_queues if media_type == "video" else self.screen_share_audio_queues
                if pc_id and queues.get(pc_id) is out_q: queues.pop(pc_id, None)
            try: client_sock.close()
            except OSError: pass
            if pc_id: self._log_event("STUDENT_SCREEN_SHARE_DISCONNECTED", pc_id=pc_id, media=media_type)

    def send_command(self, pc_id: str, command: str, payload: Optional[dict] = None, *, allow_udp_fallback: bool = True) -> Optional[str]:
        if command in {"SHUTDOWN", "RESTART"}:
            allow_udp_fallback = False
        with self.lock:
            client = self.clients.get(pc_id)
            if not client:
                return None
            control_sock = client.control_sock
            control_send_lock = client.control_send_lock
            peer_ip = client.peer_ip
            control_generation = client.control_generation
        cmd_id = f"cmd-{uuid.uuid4().hex[:12]}"
        msg = {"type": "command", "pc_id": pc_id, "command": command, "cmd_id": cmd_id}
        if payload:
            msg.update(payload)
        if control_sock:
            try:
                with control_send_lock:
                    send_json(control_sock, msg)
                with self.lock:
                    current = self.clients.get(pc_id)
                    if not current or current.control_generation != control_generation or current.control_sock is not control_sock:
                        return None
                self._log_event("command_sent", pc_id=pc_id, command=command, cmd_id=cmd_id, transport="tcp")
                return cmd_id
            except OSError:
                pass
        if allow_udp_fallback and peer_ip:
            try:
                if not self._send_udp_fallback(msg, peer_ip):
                    return None
                self._log_event("command_sent", pc_id=pc_id, command=command, cmd_id=cmd_id, transport="udp_fallback")
                return cmd_id
            except OSError:
                return None
        return None

    def shutdown_targets(self, targets: list[str]) -> list[str]:
        with self.lock:
            eligible = [
                pc_id for pc_id in targets
                if pc_id in self.clients and self.clients[pc_id].online and self.clients[pc_id].control_sock is not None
            ]
        for pc_id in eligible:
            self.send_command(pc_id, "SHUTDOWN", allow_udp_fallback=False)
            self._put_bounded(self.status_queue, pc_id)
        return eligible

    def restart_targets(self, targets: list[str]) -> list[str]:
        with self.lock:
            eligible = [
                pc_id for pc_id in targets
                if pc_id in self.clients and self.clients[pc_id].online and self.clients[pc_id].control_sock is not None
            ]
        for pc_id in eligible:
            self.send_command(pc_id, "RESTART", allow_udp_fallback=False)
            self._put_bounded(self.status_queue, pc_id)
        return eligible

    def lock_targets(self, targets: list[str], signout: bool = False) -> None:
        payload = {"lock_mode": "signout" if signout else "temporary"}
        if signout:
            payload["pause_ts"] = time.time()
        if (not signout) and bool(self.settings.enable_timer_pause_on_temp_lock):
            self._pause_timers_for_targets(targets)
            now = time.time()
            with self.lock:
                for pc_id in targets:
                    c = self.clients.get(pc_id)
                    if c and c.current_user and c.session_timer and (not c.session_timer.paused):
                        c.session_timer.paused = True
                        c.session_timer.paused_at_ts = now
            for pc_id in targets:
                self.send_command(pc_id, "PAUSE_TIMER", {"timer_id": "session", "pause_ts": now})
        for pc_id in targets:
            with self.lock:
                client = self.clients.get(pc_id)
                if client:
                    self._apply_lock_reason(client, "signout" if signout else "manual")
            cmd_id = self.send_command(pc_id, "LOCK_NOW", payload)
            if cmd_id:
                self._put_bounded(self.status_queue, pc_id)
            if signout:
                should_stop = False
                with self.lock:
                    client = self.clients.get(pc_id)
                    if not client:
                        continue
                    if client.current_user:
                        self.auth_db.close_active_session(pc_id, status="admin_signout")
                        should_stop = True
                    pause_ts = payload.get("pause_ts")
                    effective_pause_ts = float(pause_ts) if isinstance(pause_ts, (int, float)) else time.time()
                    if client.session_timer and (not client.session_timer.paused):
                        client.session_timer.paused = True
                        client.session_timer.paused_at_ts = effective_pause_ts
                    client.current_user = ""
                    client.year_section = ""
                    client.student_number = ""
                    client.auth_session_id = None
                    client.interrupted = False
                    client.interrupted_remaining_s = 0
                    client.interrupted_until_ts = None
                    client.interrupted_student_number = ""
                    self.active_sessions_by_pc_id[pc_id] = {
                        "session_id": 0,
                        "full_name": "",
                        "student_number": "",
                        "year_section": "",
                        "login_ts": float(client.session_timer.start_ts) if client.session_timer else time.time(),
                        "session_timer": self._session_timer_payload(client.session_timer),
                    }
                    self._clear_conversation_for_pc(pc_id)
                if should_stop:
                    self._stop_recording(pc_id, status="admin_signout")
                self._put_bounded(self.status_queue, pc_id)
        self._persist_session_timers()

    def unlock_targets(self, targets: list[str]) -> None:
        if bool(self.settings.enable_timer_pause_on_temp_lock):
            self._resume_timers_for_targets(targets)
            now = time.time()
            with self.lock:
                for pc_id in targets:
                    c = self.clients.get(pc_id)
                    if c and c.session_timer and c.session_timer.paused:
                        if c.session_timer.paused_at_ts is not None:
                            c.session_timer.paused_accum_ms += max(0, int((now - c.session_timer.paused_at_ts) * 1000))
                        c.session_timer.paused = False
                        c.session_timer.paused_at_ts = None
            for pc_id in targets:
                self.send_command(pc_id, "RESUME_TIMER", {"timer_id": "session", "resume_ts": now})
        for pc_id in targets:
            with self.lock:
                client = self.clients.get(pc_id)
                if client and client.lock_reason == "signout":
                    continue
                if client:
                    self._apply_lock_reason(client, None, clear=True)
            cmd_id = self.send_command(pc_id, "UNLOCK_NOW")
            if cmd_id:
                self._put_bounded(self.status_queue, pc_id)

    def set_timer(self, targets: list[str], duration_ms: int, warning_ms: int) -> None:
        timer_id = f"timer-{int(time.time() * 1000)}"
        timer = TimerState(timer_id=timer_id, targets=targets, start_ts=time.time(), duration_ms=duration_ms, warning_ms=warning_ms)
        with self.lock:
            self.timers[timer_id] = timer
        payload = {
            "timer_id": timer_id,
            "targets": targets,
            "start_ts": timer.start_ts,
            "duration_ms": duration_ms,
            "warning_ms": warning_ms,
            "action": "LOCK",
        }
        for target in targets:
            self.send_command(target, "SET_TIMER", payload)
        self._persist_timers()

    def extend_timer(self, targets: list[str], extra_ms: int) -> None:
        if extra_ms == 0:
            return
        target_set = set(targets)
        affected_pairs: list[tuple[str, str]] = []
        with self.lock:
            timers = list(self.timers.values())
        for timer in timers:
            if not timer.active:
                continue
            affected = [pc for pc in timer.targets if pc in target_set]
            if not affected:
                continue
            timer.duration_ms = max(1000, int(timer.duration_ms + extra_ms))
            payload = {"timer_id": timer.timer_id, "extra_ms": int(extra_ms)}
            for pc_id in affected:
                self.send_command(pc_id, "EXTEND_TIMER", payload)
                affected_pairs.append((timer.timer_id, pc_id))
        with self.lock:
            for timer_id, pc_id in affected_pairs:
                key = (timer_id, pc_id)
                self.session_extension_ms_by_timer_pc[key] = int(self.session_extension_ms_by_timer_pc.get(key, 0)) + int(extra_ms)
            for pc_id in targets:
                client = self.clients.get(pc_id)
                if client and client.current_user and client.session_timer:
                    client.session_timer.duration_ms = max(0, int(client.session_timer.duration_ms + extra_ms))
                    if pc_id in self.active_sessions_by_pc_id:
                        self.active_sessions_by_pc_id[pc_id]["session_timer"] = self._session_timer_payload(client.session_timer)
        affected_pc_ids = {pc_id for _, pc_id in affected_pairs}
        for pc_id in targets:
            if pc_id in affected_pc_ids:
                continue
            self.send_command(pc_id, "EXTEND_TIMER", {"timer_id": "session", "extra_ms": int(extra_ms)})
        self._persist_timers()
        self._persist_session_timers()

    def cancel_timer(self, targets: list[str]) -> None:
        target_set = set(targets)
        rollback_pairs: list[tuple[str, str]] = []
        with self.lock:
            timers = list(self.timers.values())
        for timer in timers:
            if not timer.active:
                continue
            affected = [pc for pc in timer.targets if pc in target_set]
            if not affected:
                continue
            timer.active = False
            payload = {"timer_id": timer.timer_id}
            for pc_id in affected:
                self.send_command(pc_id, "CANCEL_TIMER", payload)
                rollback_pairs.append((timer.timer_id, pc_id))
        with self.lock:
            for key in rollback_pairs:
                rollback_ms = int(self.session_extension_ms_by_timer_pc.pop(key, 0))
                if rollback_ms == 0:
                    continue
                pc_id = key[1]
                client = self.clients.get(pc_id)
                if client and client.session_timer:
                    client.session_timer.duration_ms = max(0, int(client.session_timer.duration_ms - rollback_ms))
        self._persist_timers()

    def _active_timers(self, pc_id: str) -> list[dict]:
        now = time.time()
        data: list[dict] = []
        with self.lock:
            timers = list(self.timers.values())
        for timer in timers:
            if not timer.active or pc_id not in timer.targets:
                continue
            remaining_ms = self._timer_remaining_ms(timer, now)
            data.append({
                "timer_id": timer.timer_id,
                "targets": timer.targets,
                "start_ts": timer.start_ts,
                "duration_ms": timer.duration_ms,
                "warning_ms": timer.warning_ms,
                "remaining_ms": remaining_ms,
                "paused": bool(timer.paused),
                "paused_at_ts": timer.paused_at_ts,
                "paused_accum_ms": int(timer.paused_accum_ms),
                "action": "LOCK",
            })
        return data


class TeacherDeployUI:
    STATUS_COLORS = STATUS_COLORS
    DISPLAY_STATUS = {
        "ONLINE (HEALTHY)": "Healthy",
        "ONLINE (DEGRADED: VIDEO DELAY)": "Video Delay",
        "ONLINE (SENSOR STALE)": "Sensor Stale",
        "ONLINE (SENSOR DATA WAITING)": "Waiting Data",
        "LOCKED (MANUAL/TIMER)": "Locked",
        "OFFLINE (NO HEARTBEAT)": "Offline",
    }
    TILE_BG = CARD_BG
    TILE_BORDER = BORDER_SUBTLE
    SELECTED_BORDER = ESSU_PRIMARY
    FONT_FAMILY = "Arial"
    MIN_TILE_REPAINT_S = 0.18
    MAX_TILE_UPDATES_PER_DRAIN = 12

    def __init__(self, server: TeacherDeployServer) -> None:
        ctk.set_appearance_mode(server.settings.theme_mode if server.settings.theme_mode in {"light", "dark", "system"} else "light")
        self.server = server
        self.root = ctk.CTk()
        self.root.title("IoT-Based Smart Laboratory Management System")
        screen_w = max(1024, int(self.root.winfo_screenwidth()))
        screen_h = max(768, int(self.root.winfo_screenheight()))
        init_w = min(screen_w, int(screen_w * 0.95))
        init_h = min(screen_h, int(screen_h * 0.90))
        self.root.geometry(f"{init_w}x{init_h}+20+20")
        self.root.after(0, lambda: self.root.state("zoomed"))
        self.root.configure(fg_color=UI_BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close_requested)
        self.root.report_callback_exception = self._report_tk_callback_exception

        self.selected_pc: Optional[str] = None
        self.tiles: dict[str, dict[str, object]] = {}
        self.target_checks: dict[str, ctk.CTkCheckBox] = {}
        self.extended_timer_ms_by_pc: dict[str, int] = {}
        self._font_cache: dict[int, ImageFont.ImageFont] = {}
        self._last_tile_repaint_ts: dict[str, float] = {}
        self._last_large_repaint_ts: dict[str, float] = {}
        self._pending_pc_updates: set[str] = set()
        self._drain_rr_index: int = 0
        self.chat_sidebar_window: Optional[ctk.CTkToplevel] = None
        self.chat_sidebar_body: Optional[ctk.CTkScrollableFrame] = None
        self._runtime_notice_default = "Runtime monitor active."
        self._runtime_notice_until_ts = 0.0
        self.student_management_panel = StudentManagementPanel(
            parent=self.root,
            auth_db=self.server.auth_db,
            theme_palette_provider=self._theme_palette,
            font_family=self.FONT_FAMILY,
        )
        self.reservation_management_panel = ReservationManagementPanel(
            parent=self.root,
            reservation_manager=self.server.reservation_manager,
            auth_db=self.server.auth_db,
            pc_ids_provider=self.server.registry.list_pc_ids,
            theme_palette_provider=self._theme_palette,
            font_family=self.FONT_FAMILY,
        )

        # =========================
        # 1) TOP BAR (Fixed Height = 160) — SENSOR PANEL
        # =========================
        top = ctk.CTkFrame(self.root, height=122, fg_color=CARD_BG, border_width=1, border_color=BORDER_SUBTLE)
        self.top_bar = top
        top.pack(fill="x", padx=12, pady=(10, 6))
        top.pack_propagate(False)

        title_row = ctk.CTkFrame(top, fg_color="transparent")
        title_row.pack(fill="x", padx=16, pady=(10, 6))
        # UI POLISH ONLY
        self.top_title_label = ctk.CTkLabel(
            title_row,
            text="Selected Workstation Overview",
            font=(self.FONT_FAMILY, 17, "bold"),
            text_color=TEXT_PRIMARY
        )
        self.top_title_label.pack(side="left")
        self.settings_button = ctk.CTkButton(title_row, text="Settings", command=self._open_settings_modal, width=110, height=34, **BUTTON_NEUTRAL)
        self.settings_button.pack(side="right", padx=(8, 0))
        self.reservations_button = ctk.CTkButton(
            title_row,
            text="Reservations",
            command=self._open_reservation_management,
            width=120,
            height=34,
            **BUTTON_NEUTRAL,
        )
        self.reservations_button.pack(side="right", padx=(8, 0))
        self.chat_btn = ctk.CTkButton(
            title_row,
            text="Chat",
            command=self._open_chat_sidebar,
            width=120,
            height=34,
            **BUTTON_NEUTRAL,
        )
        self.chat_btn.pack(side="right", padx=(8, 0), after=self.reservations_button)
        self.chat_unread_dot = ctk.CTkFrame(self.chat_btn, width=10, height=10, corner_radius=5, fg_color="#DC2626")
        self.chat_unread_dot.place_forget()
        self.students_button = ctk.CTkButton(
            title_row,
            text="Students",
            command=self._open_student_management,
            width=100,
            height=34,
            **BUTTON_NEUTRAL,
        )
        self.students_button.pack(side="right", padx=(8, 0))
        self.selected_history_button = ctk.CTkButton(
            title_row,
            text="View Selected History",
            command=self._open_selected_history,
            width=170,
            height=34,
            state="disabled",
            **BUTTON_NEUTRAL,
        )
        self.selected_history_button.pack(side="right")

        sensor_row = ctk.CTkFrame(top, fg_color="transparent")
        sensor_row.pack(fill="x", padx=18, pady=(0, 12))

        self.sensor_value_labels: dict[str, ctk.CTkLabel] = {}
        self.sensor_title_labels: list[ctk.CTkLabel] = []
        self.sensor_separators: list[ctk.CTkFrame] = []
        sensor_fields = [
            ("pc", "PC", "--"),
            ("temp", "Temp", "--"),
            ("rpm", "RPM", "--"),
            ("cpu", "CPU", "--"),
            ("ram", "RAM", "--"),
            ("disk", "Disk", "--"),
            ("uptime", "Uptime", "--"),
            ("user", "Current User", "--"),
            ("student_number", "Student No.", "--"),
            ("session_left", "Session Left", "--"),
            ("status", "Status", "--"),
            ("timer_extended", "Extended Timer", "00:00"),
            ("sensor_age", "Last sensor update", "--"),
            ("system", "System Online", "--"),
        ]
        group_breaks = {"uptime", "session_left"}
        for key, title, initial in sensor_fields:
            block = ctk.CTkFrame(sensor_row, fg_color="transparent")
            block.pack(side="left", padx=22)
            title_label = ctk.CTkLabel(block, text=title, font=(self.FONT_FAMILY, 12, "bold"), text_color=TEXT_SECONDARY)
            title_label.pack(anchor="w")
            self.sensor_title_labels.append(title_label)
            value_label = ctk.CTkLabel(block, text=initial, font=(self.FONT_FAMILY, 12), text_color=TEXT_PRIMARY)
            value_label.pack(anchor="w")
            self.sensor_value_labels[key] = value_label
            if key in group_breaks:
                sep = ctk.CTkFrame(sensor_row, width=1, height=36, fg_color=BORDER_SUBTLE)
                sep.pack(side="left", padx=(2, 12), pady=(2, 0))
                self.sensor_separators.append(sep)

        # =========================
        # 2) MIDDLE (ONLY EXPANDABLE AREA)
        # =========================
        middle = ctk.CTkFrame(self.root, fg_color="transparent")
        self.middle = middle
        self.middle_panel = middle
        middle.pack(fill="both", expand=True, padx=8, pady=6)

        # LEFT SIDE — MAIN VIDEO
        self.large_view_host = ctk.CTkFrame(middle, fg_color="transparent")
        self.large_view_host.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        self.large_view_host.pack_propagate(False)
        self.large_view = ctk.CTkLabel(
            self.large_view_host,
            text="Select a workstation to view its live feed.",
            anchor="center",
            justify="center",
            wraplength=520,
            font=(self.FONT_FAMILY, 18, "bold"),
            fg_color="transparent",
            text_color=TEXT_SECONDARY,
            corner_radius=8,
        )
        self.large_view.pack(fill="both", expand=True)

        # RIGHT SIDE — PREVIEW GRID (Fixed Width 400)
        self.grid_scroll = ctk.CTkScrollableFrame(middle, width=400, fg_color="transparent", border_width=1, border_color=BORDER_SUBTLE)
        self.grid_scroll.pack(side="right", fill="y", padx=(0, 6), pady=6)
        self.grid_scroll.pack_propagate(False)

        # =========================
        # 3) BOTTOM BAR (Fixed Height = 160) — UNCHANGED
        # =========================
        bottom = ctk.CTkFrame(self.root, height=86, fg_color=CARD_BG, border_width=1, border_color=BORDER_SUBTLE)
        self.bottom_bar = bottom
        bottom.pack(fill="x", padx=10, pady=(6, 8))
        bottom.pack_propagate(False)

        self.targets_frame = ctk.CTkScrollableFrame(bottom, width=450, fg_color=CARD_BG)
        self.targets_frame.pack(side="left", fill="both", expand=False, padx=8, pady=10)

        self.left_controls_frame = ctk.CTkFrame(bottom, fg_color=CARD_BG)
        self.left_controls_frame.pack(side="left", fill="both", expand=True, padx=12, pady=10)

        self.right_controls_frame = ctk.CTkFrame(bottom, width=190, fg_color=CARD_BG)
        self.right_controls_frame.pack(side="right", fill="y", expand=False, padx=(4, 10), pady=10)
        self.right_controls_frame.pack_propagate(False)

        action_group = ctk.CTkFrame(self.left_controls_frame, fg_color="transparent")
        action_group.pack(side="left", padx=(8, 16))
        self.lock_mode_label = ctk.CTkLabel(action_group, text="Action Mode", font=(self.FONT_FAMILY, 12, "bold"), text_color=TEXT_SECONDARY)
        self.lock_mode_label.pack(side="left", padx=(0, 6))
        self.lock_mode_var = ctk.StringVar(value="Temporary Lock")
        self.lock_mode_menu = ctk.CTkOptionMenu(
            action_group,
            variable=self.lock_mode_var,
            values=["Temporary Lock", "Lock + Sign Out", "Shutdown", "Restart"],
            width=178,
            height=34,
        )
        self.lock_mode_menu.pack(side="left", padx=4)
        self.lock_btn = ctk.CTkButton(action_group, text="Apply Lock Mode", command=self._lock_targets, width=150, height=34, **BUTTON_PRIMARY)
        self.lock_btn.pack(side="left", padx=4)
        self.unlock_btn = ctk.CTkButton(action_group, text="Unlock", command=self._unlock_targets, width=110, height=34, **BUTTON_NEUTRAL)
        self.unlock_btn.pack(side="left", padx=4)
        self.lock_mode_var.trace_add("write", lambda *_args: self._refresh_control_buttons())

        timer_group = ctk.CTkFrame(self.left_controls_frame, fg_color="transparent")
        timer_group.pack(side="left", padx=(12, 6))
        self.timer_title_label = ctk.CTkLabel(timer_group, text="Extend Session", font=(self.FONT_FAMILY, 12, "bold"), text_color=TEXT_SECONDARY)
        self.timer_title_label.pack(side="left", padx=(0, 8))
        self.timer_entry = ctk.CTkEntry(timer_group, placeholder_text="Minutes", width=92, height=34, fg_color="#FAFAFA", border_color=BORDER_SUBTLE, text_color=TEXT_PRIMARY)
        self.timer_entry.pack(side="left", padx=4)
        self.extend_timer_btn = ctk.CTkButton(timer_group, text="Extend", command=self._extend_timer, width=96, height=34, **BUTTON_WARNING)
        self.extend_timer_btn.pack(side="left", padx=4)
        self.cancel_timer_btn = ctk.CTkButton(timer_group, text="Cancel", command=self._cancel_timer, width=96, height=34, **BUTTON_NEUTRAL)
        self.cancel_timer_btn.pack(side="left", padx=4)
        self.screen_share_btn = ctk.CTkButton(timer_group, text="Share Screen", command=self._toggle_screen_share, width=120, height=34, **BUTTON_PRIMARY)
        self.screen_share_btn.pack(side="left", padx=(12, 4))

        self.dashboard_hint_title = ctk.CTkLabel(
            self.right_controls_frame,
            text="Quick Guide",
            font=(self.FONT_FAMILY, 12, "bold"),
            text_color=TEXT_PRIMARY,
            anchor="w",
            justify="left",
        )
        self.dashboard_hint_title.pack(fill="x", padx=10, pady=(6, 2))
        self.dashboard_hint_label = ctk.CTkLabel(
            self.right_controls_frame,
            text="Select a tile to focus it or tick one or more targets.",
            font=(self.FONT_FAMILY, 11),
            text_color=TEXT_SECONDARY,
            anchor="w",
            justify="left",
            wraplength=160,
        )
        self.dashboard_hint_label.pack(fill="x", padx=10, pady=(0, 6))
        self.runtime_notice_label = ctk.CTkLabel(
            self.right_controls_frame,
            text=self._runtime_notice_default,
            font=(self.FONT_FAMILY, 11),
            text_color=TEXT_SECONDARY,
            anchor="w",
            justify="left",
            wraplength=160,
        )
        self.runtime_notice_label.pack(fill="x", padx=10, pady=(0, 6))

        # self.approve_ext_btn = ctk.CTkButton(self.left_controls_frame, text="Approve Extension", command=self._approve_extension_selected, width=140, **BUTTON_WARNING)
        # self.approve_ext_btn.pack(side="right", padx=4, pady=8)



        self._apply_theme_to_ui()
        self._refresh_control_buttons()
        self.root.after(100, self._drain_queues)


    def _theme_palette(self) -> dict[str, str]:
        mode = ctk.get_appearance_mode().lower()
        if mode == "dark":
            return {
                "root_bg": "#13161d",
                "card_bg": "#202631",
                "text_primary": "#f1f5ff",
                "text_secondary": DARK_TEXT_SECONDARY,
                "border": "#313949",
            }
        return {
            "root_bg": UI_BG,
            "card_bg": CARD_BG,
            "text_primary": TEXT_PRIMARY,
            "text_secondary": TEXT_SECONDARY,
            "border": BORDER_SUBTLE,
        }

    def _apply_theme_to_ui(self) -> None:
        colors = self._theme_palette()
        self.root.configure(fg_color=colors["root_bg"])
        for frame in (
            getattr(self, "top_bar", None),
            getattr(self, "bottom_bar", None),
            getattr(self, "targets_frame", None),
            getattr(self, "left_controls_frame", None),
            getattr(self, "right_controls_frame", None),
        ):
            if frame is not None:
                frame.configure(fg_color=colors["card_bg"], border_color=colors["border"])
        if hasattr(self, "grid_scroll"):
            self.grid_scroll.configure(fg_color="transparent", border_color=colors["border"])
        if hasattr(self, "large_view"):
            self.large_view.configure(fg_color="transparent", text_color=colors["text_secondary"])
        if hasattr(self, "top_title_label"):
            self.top_title_label.configure(text_color=colors["text_primary"])
        if hasattr(self, "lock_mode_label"):
            self.lock_mode_label.configure(text_color=colors["text_secondary"])
        if hasattr(self, "timer_title_label"):
            self.timer_title_label.configure(text_color=colors["text_secondary"])
        if hasattr(self, "dashboard_hint_title"):
            self.dashboard_hint_title.configure(text_color=colors["text_primary"])
        if hasattr(self, "dashboard_hint_label"):
            self.dashboard_hint_label.configure(text_color=colors["text_secondary"])
        if hasattr(self, "runtime_notice_label") and (time.time() >= getattr(self, "_runtime_notice_until_ts", 0.0)):
            self.runtime_notice_label.configure(text_color=colors["text_secondary"])
        if hasattr(self, "timer_entry"):
            entry_bg = "#262d38" if ctk.get_appearance_mode().lower() == "dark" else "#FAFAFA"
            self.timer_entry.configure(fg_color=entry_bg, text_color=colors["text_primary"], border_color=colors["border"])
        for title_label in getattr(self, "sensor_title_labels", []):
            title_label.configure(text_color=colors["text_secondary"])
        for value_label in self.sensor_value_labels.values():
            value_label.configure(text_color=colors["text_primary"])
        for cb in self.target_checks.values():
            cb.configure(text_color=colors["text_primary"], border_color=colors["border"])
        for tile in self.tiles.values():
            tile_widget = tile.get("tile")
            preview_widget = tile.get("preview")
            indicator = tile.get("indicator")
            if isinstance(tile_widget, ctk.CTkFrame):
                tile_widget.configure(fg_color="transparent", border_color=colors["border"])
            if isinstance(preview_widget, ctk.CTkLabel):
                preview_widget.configure(fg_color="transparent", text_color=colors["text_secondary"])
            if isinstance(indicator, ctk.CTkLabel):
                indicator.configure(fg_color="transparent")

    def _apply_theme_to_toplevel(self, win: ctk.CTkToplevel) -> dict[str, str]:
        colors = self._theme_palette()
        win.configure(fg_color=colors["root_bg"])
        return colors

    def _status_badge_text(self, full_status: str) -> str:
        badge_map = {
            "ONLINE (HEALTHY)": "HEALTHY",
            "ONLINE (DEGRADED: VIDEO DELAY)": "DEGRADED",
            "ONLINE (SENSOR STALE)": "SENSOR STALE",
            "ONLINE (SENSOR DATA WAITING)": "WAITING",
            "LOCKED (MANUAL/TIMER)": "LOCKED",
            "OFFLINE (NO HEARTBEAT)": "OFFLINE",
        }
        return badge_map.get(full_status, self._display_status(full_status).upper())


    def _configure_if_changed(self, widget, **kwargs) -> None:
        changed = {}
        for key, value in kwargs.items():
            try:
                current = widget.cget(key)
            except Exception:
                current = None
            if current != value:
                changed[key] = value
        if changed:
            widget.configure(**changed)

    def _set_runtime_notice(self, text: str, text_color: Optional[str] = None, hold_s: float = 0.0) -> None:
        color = text_color or self._theme_palette()["text_secondary"]
        if hasattr(self, "runtime_notice_label"):
            self._configure_if_changed(self.runtime_notice_label, text=text, text_color=color)
        self._runtime_notice_until_ts = (time.time() + max(0.0, float(hold_s))) if hold_s > 0 else 0.0

    def _refresh_runtime_notice(self) -> None:
        if self._runtime_notice_until_ts and time.time() >= self._runtime_notice_until_ts:
            self._runtime_notice_until_ts = 0.0
            self._set_runtime_notice(self._runtime_notice_default, self._theme_palette()["text_secondary"])

    def _report_tk_callback_exception(self, exc, val, tb) -> None:
        detail = "".join(traceback.format_exception(exc, val, tb))[-4000:]
        self.server._log_event("ui_callback_error", detail=detail)
        self._set_runtime_notice("Dashboard recovered after a UI callback issue.", ESSU_WARNING, hold_s=20.0)

    def _on_close_requested(self) -> None:
        if messagebox.askyesno(
            "Exit Admin Dashboard",
            "Closing this window also stops the lab server for the computer lab.\n\nDo you want to exit?",
        ):
            self.server.stop_screen_share()
            self.root.destroy()



    def _font(self, size: int) -> ImageFont.ImageFont:
        cached = self._font_cache.get(size)
        if cached is not None:
            return cached
        for name in ("DejaVuSans-Bold.ttf", "arial.ttf"):
            try:
                font = ImageFont.truetype(name, size)
                self._font_cache[size] = font
                return font
            except OSError:
                continue
        font = ImageFont.load_default()
        self._font_cache[size] = font
        return font

    def _apply_preview_overlay(self, image: Image.Image, pc_id: str) -> Image.Image:
        canvas = image.convert("RGBA")
        draw = ImageDraw.Draw(canvas)
        font = self._font(50)
        shadow_offset = 2
        x = max(8, canvas.width - 210)
        y = 12
        draw.text((x + shadow_offset, y + shadow_offset), pc_id, font=font, fill=(0, 0, 0, 120))
        draw.text((x, y), pc_id, font=font, fill=(255, 255, 255, 220))
        client = self.server.clients.get(pc_id)
        if client and client.current_user:
            second_y = y + 36
            draw.text((x + shadow_offset, second_y + shadow_offset), client.current_user, font=font, fill=(0, 0, 0, 120))
            draw.text((x, second_y), client.current_user, font=font, fill=(255, 255, 255, 220))
        return canvas.convert("RGB")

    def _apply_preview_status_dot(self, image: Image.Image, color: str) -> Image.Image:
        canvas = image.convert("RGBA")
        draw = ImageDraw.Draw(canvas)
        dot_color = str(color or "").strip()
        if not (dot_color.startswith("#") and len(dot_color) == 7):
            dot_color = "#FFFFFF"
        r = 6
        cx = 18
        cy = 18
        fill = (
            int(dot_color[1:3], 16),
            int(dot_color[3:5], 16),
            int(dot_color[5:7], 16),
            235,
        )
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill)
        return canvas.convert("RGB")

    def _apply_main_overlay(self, image: Image.Image, pc_id: str) -> Image.Image:
        status, _ = self._status_for_pc(pc_id)
        canvas = image.convert("RGBA")
        draw = ImageDraw.Draw(canvas)
        text = f"{pc_id}  |  {self.DISPLAY_STATUS.get(status, status)}"
        font = self._font(24)
        x = 16
        y = 14
        draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0, 120))
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 230))
        return canvas.convert("RGB")


    def _grid_position(self, idx: int) -> tuple[int, int]:
        return idx % 4, idx // 4

    def _ensure_tile(self, pc_id: str) -> None:
        if pc_id in self.tiles:
            return
        colors = self._theme_palette()
        tile = ctk.CTkFrame(
            self.grid_scroll,
            width=260,
            height=180,
            fg_color="transparent",
            border_width=1,
            border_color=colors["border"],
            corner_radius=8,
        )
        tile.grid_propagate(False)
        preview = ctk.CTkLabel(tile, text=pc_id, fg_color="transparent", text_color=colors["text_secondary"])
        preview.pack(fill="both", expand=True, padx=6, pady=6)
        indicator = ctk.CTkLabel(tile, text="●", font=("Arial", 14, "bold"), text_color=colors["text_secondary"], fg_color="transparent")
        indicator.place(x=8, y=6)
        indicator.configure(text="")
        tile.bind("<Button-1>", lambda _e, pid=pc_id: self._select_pc(pid))
        preview.bind("<Button-1>", lambda _e, pid=pc_id: self._select_pc(pid))

        idx = len(self.tiles)
        row, col = self._grid_position(idx)
        tile.grid(row=row, column=col, padx=10, pady=10)
        self.tiles[pc_id] = {"tile": tile, "preview": preview, "indicator": indicator}

        cb = ctk.CTkCheckBox(
            self.targets_frame,
            text=f"{pc_id}",
            text_color=colors["text_primary"],
            border_color=colors["border"],
            command=self._refresh_control_buttons,
        )
        cb_index = len(self.target_checks)
        cb_row = cb_index % 2
        cb_col = cb_index // 2
        cb.grid(row=cb_row, column=cb_col, padx=8, pady=4, sticky="w")
        self.target_checks[pc_id] = cb

    def _select_pc(self, pc_id: str) -> None:
        self.selected_pc = pc_id
        self.selected_history_button.configure(state="normal")
        self._update_sensor_panel(pc_id)
        self._refresh_control_buttons()

    def _selected_targets(self) -> list[str]:
        targets = [pc for pc, cb in self.target_checks.items() if cb.get() == 1]
        if not targets and self.selected_pc:
            targets = [self.selected_pc]
        return targets

    def _logged_in_targets(self, targets: list[str]) -> list[str]:
        with self.server.lock:
            return [pc for pc in targets if (pc in self.server.clients and bool(self.server.clients[pc].current_user))]

    def _temporary_locked_targets(self, targets: list[str]) -> list[str]:
        with self.server.lock:
            return [pc for pc in targets if (pc in self.server.clients and self.server.clients[pc].lock_reason == "manual")]

    def _online_targets(self, targets: list[str]) -> list[str]:
        with self.server.lock:
            return [pc for pc in targets if (pc in self.server.clients and self.server.clients[pc].online)]

    def _refresh_control_buttons(self) -> None:
        targets = self._selected_targets()
        logged = self._logged_in_targets(targets)
        temp_locked = self._temporary_locked_targets(targets)
        online = self._online_targets(targets)
        mode = self.lock_mode_var.get()

        if mode in {"Shutdown", "Restart"}:
            can_lock = bool(online)
            primary_text = "Shutdown PCs" if mode == "Shutdown" else "Restart PCs"
        else:
            can_lock = bool(logged)
            primary_text = "Apply Lock Mode"
        can_unlock = bool(temp_locked)
        can_extend = len(logged) == 1
        selected_ext = int(self.extended_timer_ms_by_pc.get(logged[0], 0)) if len(logged) == 1 else 0
        can_cancel = len(logged) == 1 and selected_ext > 0

        self._configure_if_changed(self.lock_btn, text=primary_text, state="normal" if can_lock else "disabled")
        self._configure_if_changed(self.unlock_btn, state="normal" if can_unlock else "disabled")
        self._configure_if_changed(self.extend_timer_btn, state="normal" if can_extend else "disabled")
        self._configure_if_changed(self.cancel_timer_btn, state="normal" if can_cancel else "disabled")
        sharing = bool(self.server.screen_share_session_id)
        eligible = bool(self.server.eligible_screen_share_targets())
        self._configure_if_changed(self.screen_share_btn, text="Stop Sharing" if sharing else "Share Screen", state="normal" if sharing or eligible else "disabled")
        with self.server.lock:
            has_unread = any(int(convo.get("unread", 0)) > 0 for convo in self.server.conversations.values())
        if has_unread:
            self.chat_unread_dot.place(relx=1.0, rely=0.0, x=-7, y=5, anchor="ne")
        else:
            self.chat_unread_dot.place_forget()

        # can_ext_approve = bool(self.server.settings.enable_extension_requests) and len(logged) == 1
        # self._configure_if_changed(self.approve_ext_btn, state="normal" if can_ext_approve else "disabled")

        if bool(self.server.settings.enable_session_messaging):
            if self.chat_btn.winfo_manager() != "pack":
                self.chat_btn.pack(side="right", padx=(8, 0), after=self.reservations_button)
            self._configure_if_changed(self.chat_btn, state="normal")
        else:
            if self.chat_btn.winfo_manager():
                self.chat_btn.pack_forget()
            if self.chat_sidebar_window is not None and self.chat_sidebar_window.winfo_exists():
                self.chat_sidebar_window.destroy()
            self.chat_sidebar_window = None
            self.chat_sidebar_body = None

    def _toggle_screen_share(self) -> None:
        if self.server.screen_share_session_id:
            self.server.stop_screen_share()
            self._set_runtime_notice("Screen sharing stopped.")
        elif self.server.start_screen_share():
            self._set_runtime_notice("Screen sharing active.")
        else:
            self._set_runtime_notice("No connected students are available for screen sharing.", ESSU_WARNING, hold_s=8.0)
        self._refresh_control_buttons()

    def _lock_targets(self) -> None:
        mode = self.lock_mode_var.get()
        if mode == "Shutdown":
            self._shutdown_targets()
            return
        if mode == "Restart":
            self._restart_targets()
            return
        targets = self._selected_targets()
        logged_targets = self._logged_in_targets(targets)
        if not logged_targets:
            return
        signout = mode == "Lock + Sign Out"
        self.server.lock_targets(logged_targets, signout=signout)
        self._refresh_control_buttons()

    def _approve_extension_selected(self) -> None:
        if not bool(self.server.settings.enable_extension_requests):
            return
        targets = self._selected_targets()
        if len(targets) != 1:
            return
        pc_id = targets[0]
        req_id = None
        with self.server.lock:
            for k, v in reversed(list(self.server.extension_requests.items())):
                if v.get("pc_id") == pc_id and v.get("status") in {"requested", "teacher_sent"}:
                    req_id = k
                    break
        if req_id:
            self.server.respond_extension_request(req_id, approved=True)

    def _unlock_targets(self) -> None:
        targets = self._selected_targets()
        temp_locked = self._temporary_locked_targets(targets)
        if not temp_locked:
            return
        self.server.unlock_targets(temp_locked)
        self._refresh_control_buttons()

    def _shutdown_targets(self) -> None:
        targets = self._selected_targets()
        online_targets = self._online_targets(targets)
        if not online_targets:
            return
        if not messagebox.askyesno("Shutdown PCs", f"Shutdown the selected workstation(s): {', '.join(online_targets)}?"):
            return
        self.server.shutdown_targets(online_targets)
        self._refresh_control_buttons()

    def _restart_targets(self) -> None:
        targets = self._selected_targets()
        online_targets = self._online_targets(targets)
        if not online_targets:
            return
        if not messagebox.askyesno("Restart PCs", f"Restart the selected workstation(s): {', '.join(online_targets)}?"):
            return
        self.server.restart_targets(online_targets)
        self._refresh_control_buttons()

    def _timer_minutes_from_input(self) -> Optional[float]:
        try:
            minutes = float(self.timer_entry.get())
        except ValueError:
            return None
        if minutes <= 0:
            return None
        return minutes

    def _extend_timer(self) -> None:
        targets = self._selected_targets()
        logged_targets = self._logged_in_targets(targets)
        if len(logged_targets) != 1:
            return
        minutes = self._timer_minutes_from_input()
        if minutes is None:
            return
        pc_id = logged_targets[0]
        extra_ms = int(minutes * 60_000)
        self.server.extend_timer([pc_id], extra_ms)
        self.extended_timer_ms_by_pc[pc_id] = int(self.extended_timer_ms_by_pc.get(pc_id, 0)) + extra_ms
        self._refresh_control_buttons()

    def _cancel_timer(self) -> None:
        targets = self._selected_targets()
        logged_targets = self._logged_in_targets(targets)
        if len(logged_targets) != 1:
            return
        pc_id = logged_targets[0]
        extended_ms = int(self.extended_timer_ms_by_pc.get(pc_id, 0))
        if extended_ms <= 0:
            self._refresh_control_buttons()
            return
        self.server.extend_timer([pc_id], -extended_ms)
        self.extended_timer_ms_by_pc[pc_id] = 0
        self._refresh_control_buttons()

    def _chat_sidebar_rows(self) -> list[dict]:
        with self.server.lock:
            rows = []
            for pc_id, client in self.server.clients.items():
                convo = self.server.conversations.get(pc_id, {})
                rows.append({
                    "pc_id": pc_id,
                    "student_name": str(client.current_user or ""),
                    "unread": int(convo.get("unread", 0)),
                    "last_ts": float(convo.get("last_ts", 0.0) or 0.0),
                })
        rows.sort(key=lambda item: (item["unread"] > 0, item["last_ts"], item["pc_id"]), reverse=True)
        return rows

    def _open_chat_sidebar(self) -> None:
        if not bool(self.server.settings.enable_session_messaging):
            return
        # Opening the dashboard chat is the read lifecycle for its single badge.
        with self.server.lock:
            for convo in self.server.conversations.values():
                convo["unread"] = 0
        self.server._log_event("CHAT_UNREAD_CLEARED")
        self._refresh_control_buttons()
        if self.chat_sidebar_window is not None and self.chat_sidebar_window.winfo_exists():
            self.chat_sidebar_window.lift()
            try:
                self.chat_sidebar_window.attributes("-topmost", True)
                self.chat_sidebar_window.after(10, lambda: self.chat_sidebar_window.attributes("-topmost", False))
            except Exception:
                pass
            self.chat_sidebar_window.focus_set()
            return

        colors = self._theme_palette()
        win = ctk.CTkToplevel(self.root)
        win.title("Chat")
        win.geometry("360x620")
        win.configure(fg_color=colors["card_bg"])
        self.chat_sidebar_window = win
        win.lift()
        try:
            win.attributes("-topmost", True)
            win.after(10, lambda: win.attributes("-topmost", False))
        except Exception:
            pass
        win.focus_set()

        header = ctk.CTkFrame(win, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(12, 8))
        ctk.CTkLabel(header, text="Session Chat", font=(self.FONT_FAMILY, 16, "bold"), text_color=colors["text_primary"]).pack(side="left")

        body = ctk.CTkScrollableFrame(win, fg_color=colors["card_bg"], border_width=1, border_color=colors["border"])
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.chat_sidebar_body = body

        def render_rows() -> None:
            if self.chat_sidebar_body is None or not self.chat_sidebar_body.winfo_exists():
                return
            for child in self.chat_sidebar_body.winfo_children():
                child.destroy()
            rows = self._chat_sidebar_rows()
            if not rows:
                ctk.CTkLabel(self.chat_sidebar_body, text="No connected workstations.", text_color=colors["text_secondary"]).pack(anchor="w", padx=10, pady=10)
                return
            for row in rows:
                pc_id = str(row["pc_id"])
                item = ctk.CTkFrame(self.chat_sidebar_body, fg_color=colors["card_bg"], border_width=1, border_color=colors["border"])
                item.pack(fill="x", padx=8, pady=6)

                avatar = ctk.CTkFrame(item, width=30, height=30, corner_radius=15, fg_color="#DDE5DE")
                avatar.pack(side="left", padx=(10, 8), pady=10)
                avatar.pack_propagate(False)
                avatar_label = ctk.CTkLabel(avatar, text=pc_id[-2:], font=(self.FONT_FAMILY, 10, "bold"), text_color=TEXT_PRIMARY)
                avatar_label.pack(expand=True)

                text_wrap = ctk.CTkFrame(item, fg_color="transparent")
                text_wrap.pack(side="left", fill="x", expand=True, pady=8)
                pc_label = ctk.CTkLabel(text_wrap, text=pc_id, font=(self.FONT_FAMILY, 13, "bold"), text_color=colors["text_primary"])
                pc_label.pack(anchor="w")
                subtitle = str(row.get("student_name", "") or "No active student")
                subtitle_label = ctk.CTkLabel(text_wrap, text=subtitle, font=(self.FONT_FAMILY, 11), text_color=colors["text_secondary"])
                subtitle_label.pack(anchor="w")

                clickable_widgets = [item, avatar, avatar_label, text_wrap, pc_label, subtitle_label]
                unread = int(row.get("unread", 0))
                if unread > 0:
                    dot = ctk.CTkFrame(item, width=18, height=18, corner_radius=9, fg_color="#DC2626")
                    dot.pack(side="right", padx=(0, 10))
                    dot.pack_propagate(False)
                    dot_label = ctk.CTkLabel(dot, text=str(min(99, unread)), font=(self.FONT_FAMILY, 10, "bold"), text_color="#FFFFFF")
                    dot_label.pack(expand=True)
                    clickable_widgets.extend([dot, dot_label])

                def _open_convo(_event=None, target_pc=pc_id) -> None:
                    self._open_conversation_modal(target_pc)

                for widget in clickable_widgets:
                    widget.bind("<Button-1>", _open_convo)

        def poll() -> None:
            if self.chat_sidebar_window is None or not self.chat_sidebar_window.winfo_exists():
                return
            render_rows()
            self.chat_sidebar_window.after(900, poll)

        def on_close() -> None:
            if win.winfo_exists():
                win.destroy()
            self.chat_sidebar_window = None
            self.chat_sidebar_body = None

        win.protocol("WM_DELETE_WINDOW", on_close)
        render_rows()
        poll()

    def _open_conversation_modal(self, pc_id: str) -> None:
        colors = self._theme_palette()
        win = ctk.CTkToplevel(self.root)
        win.title(f"Conversation - {pc_id}")
        win.geometry("700x560")
        win.configure(fg_color=colors["card_bg"])
        win.lift()
        try:
            win.attributes("-topmost", True)
            win.after(10, lambda: win.attributes("-topmost", False))
        except Exception:
            pass
        win.focus_set()

        hdr = ctk.CTkFrame(win, fg_color="transparent")
        hdr.pack(fill="x", padx=12, pady=(10, 6))
        title = ctk.CTkLabel(hdr, text=f"Conversation with {pc_id}", font=(self.FONT_FAMILY, 14, "bold"), text_color=colors["text_primary"])
        title.pack(side="left")
        unread_var = ctk.StringVar(value="Unread: 0")
        ctk.CTkLabel(hdr, textvariable=unread_var, font=(self.FONT_FAMILY, 12), text_color=colors["text_secondary"]).pack(side="right")

        body = ctk.CTkScrollableFrame(win, fg_color=colors["card_bg"], border_width=1, border_color=colors["border"])
        body.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        compose = ctk.CTkFrame(win, fg_color="transparent")
        compose.pack(fill="x", padx=12, pady=(0, 10))
        entry = ctk.CTkEntry(compose, placeholder_text="Type message...")
        entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        send_btn = ctk.CTkButton(compose, text="Send", width=90, command=lambda: send_message(), **BUTTON_NEUTRAL)
        send_btn.pack(side="left")

        status_var = ctk.StringVar(value="")
        status_label = ctk.CTkLabel(win, textvariable=status_var, text_color=ESSU_ERROR, font=(self.FONT_FAMILY, 11))
        status_label.pack(anchor="w", padx=14, pady=(0, 10))

        rendered_count = 0
        empty_label: Optional[ctk.CTkLabel] = None

        def _append_message_row(msg: dict) -> None:
            ts = self._format_ts(msg.get("ts"))
            direction = str(msg.get("direction", ""))
            if direction == "student_to_teacher":
                prefix = "Student"
                anchor = "w"
            else:
                prefix = "Teacher"
                anchor = "e"
            text = str(msg.get("text", ""))
            row = ctk.CTkFrame(body, fg_color="transparent")
            row.pack(fill="x", padx=8, pady=4)
            bubble = ctk.CTkFrame(row, fg_color="#1f6aa5", corner_radius=8, border_width=0)
            bubble.pack(anchor=anchor, padx=10, pady=6)
            ctk.CTkLabel(bubble, text=f"{prefix} - {ts}", font=(self.FONT_FAMILY, 11, "bold"), text_color="#FFFFFF").pack(anchor="w", padx=10, pady=(6, 0))
            ctk.CTkLabel(bubble, text=text, font=(self.FONT_FAMILY, 12), text_color="#FFFFFF", justify="left", wraplength=320).pack(anchor="w", padx=10, pady=(0, 6))

        def refresh(mark_read: bool = False) -> None:
            nonlocal rendered_count, empty_label
            snapshot = self.server.get_conversation_snapshot(pc_id, mark_read=mark_read)
            unread_var.set(f"Unread: {int(snapshot.get('unread', 0))}")
            messages = snapshot.get("messages", [])
            if not messages:
                if rendered_count == 0 and (empty_label is None or not empty_label.winfo_exists()):
                    empty_label = ctk.CTkLabel(body, text="No messages yet.", text_color=colors["text_secondary"])
                    empty_label.pack(anchor="w", padx=8, pady=8)
                return
            if empty_label is not None and empty_label.winfo_exists():
                empty_label.destroy()
                empty_label = None
            if rendered_count > len(messages):
                for child in body.winfo_children():
                    child.destroy()
                rendered_count = 0
            for msg in messages[rendered_count:]:
                _append_message_row(msg)
            rendered_count = len(messages)

        def send_message() -> None:
            text = entry.get().strip()
            if not text:
                return
            cmd_id = self.server.send_session_message(pc_id, text)
            if not cmd_id:
                status_var.set("Unable to send message. Ensure student session is active.")
                return
            status_var.set("")
            entry.delete(0, "end")
            refresh(mark_read=True)

        def poll() -> None:
            if not win.winfo_exists():
                return
            refresh(mark_read=True)
            win.after(750, poll)

        entry.bind("<Return>", lambda _e: send_message())
        refresh(mark_read=True)
        poll()
    def _status_for_pc(self, pc_id: str) -> tuple[str, str]:
        state = self.server.effective_states.get(pc_id)
        if state:
            return state.status_text, state.color
        return "OFFLINE (NO HEARTBEAT)", self.STATUS_COLORS["OFFLINE (NO HEARTBEAT)"]

    def _display_status(self, full_status: str) -> str:
        return self.DISPLAY_STATUS.get(full_status, full_status)

    def _render_offline_tile(self, pc_id: str) -> Image.Image:
        image = Image.new("RGB", (RUNTIME.frame_width, RUNTIME.frame_height), (245, 246, 248))
        draw = ImageDraw.Draw(image)
        title_font = self._font(42)
        subtitle_font = self._font(28)
        title = f"{pc_id}"
        subtitle = "OFFLINE"
        info = "No Live Screen"
        t_box = draw.textbbox((0, 0), title, font=title_font)
        s_box = draw.textbbox((0, 0), subtitle, font=title_font)
        i_box = draw.textbbox((0, 0), info, font=subtitle_font)
        total_h = (t_box[3]-t_box[1]) + (s_box[3]-s_box[1]) + (i_box[3]-i_box[1]) + 24
        y = (image.height - total_h) // 2
        for text_value, font, color in ((title, title_font, (31, 31, 31)), (subtitle, title_font, (178, 34, 34)), (info, subtitle_font, (110, 110, 110))):
            box = draw.textbbox((0, 0), text_value, font=font)
            w = box[2]-box[0]
            h = box[3]-box[1]
            x = (image.width - w) // 2
            draw.text((x, y), text_value, font=font, fill=color)
            y += h + 8
        return image

    def _format_ts(self, ts: Optional[float]) -> str:
        if ts is None:
            return "--"
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

    def _open_history_modal(self, title: str, records: list[dict], include_pc: bool = False) -> None:
        win = ctk.CTkToplevel(self.root)
        win.title(title)
        screen_w = max(800, int(self.root.winfo_screenwidth()))
        screen_h = max(600, int(self.root.winfo_screenheight()))
        modal_w = min(1100, int(screen_w * 0.92))
        modal_h = min(640, int(screen_h * 0.86))
        win.geometry(f"{modal_w}x{modal_h}")
        win.transient(self.root)
        win.lift()
        win.focus_force()
        win.grab_set()
        win.bind("<Escape>", lambda _e: win.destroy())
        colors = self._apply_theme_to_toplevel(win)
        self.root.update_idletasks()
        root_x = self.root.winfo_x()
        root_y = self.root.winfo_y()
        root_w = self.root.winfo_width()
        root_h = self.root.winfo_height()
        pos_x = root_x + max(0, (root_w - modal_w) // 2)
        pos_y = root_y + max(0, (root_h - modal_h) // 2)
        win.geometry(f"{modal_w}x{modal_h}+{pos_x}+{pos_y}")

        source_records = list(records)
        filtered_records = list(source_records)
        current_page = 0
        day_keys: list[str] = []
        grouped_by_day: dict[str, list[dict]] = {}
        current_visible_rows: list[dict] = []
        summary_var = ctk.StringVar(value=f"{len(source_records)} record(s) loaded. Use Play to open saved recordings.")

        header = ctk.CTkFrame(win, fg_color=colors["card_bg"], border_width=1, border_color=colors["border"])
        header.pack(fill="x", padx=10, pady=(10, 6))
        title_var = ctk.StringVar(value=title)
        ctk.CTkLabel(header, textvariable=title_var, font=(self.FONT_FAMILY, 16, "bold"), text_color=colors["text_primary"]).pack(anchor="w", padx=12, pady=(10, 4))
        # UI POLISH ONLY
        ctk.CTkLabel(
            header,
            text="Filter by student, section, or date. Saved recordings stay available in the Recording column.",
            font=(self.FONT_FAMILY, 12),
            text_color=colors["text_secondary"],
        ).pack(anchor="w", padx=12, pady=(0, 2))
        ctk.CTkLabel(
            header,
            textvariable=summary_var,
            font=(self.FONT_FAMILY, 11),
            text_color=colors["text_secondary"],
        ).pack(anchor="w", padx=12, pady=(0, 10))

        filter_bar = ctk.CTkFrame(win, fg_color=colors["card_bg"], border_width=1, border_color=colors["border"])
        filter_bar.pack(fill="x", padx=10, pady=(0, 6))
        search_var = ctk.StringVar(value="")
        section_var = ctk.StringVar(value="")
        date_var = ctk.StringVar(value="")
        page_label = ctk.CTkLabel(filter_bar, text="Page 1/1", text_color=colors["text_secondary"])

        ctk.CTkLabel(filter_bar, text="Search", text_color=colors["text_primary"]).pack(side="left", padx=(10, 4), pady=8)
        search_entry = ctk.CTkEntry(filter_bar, textvariable=search_var, width=220, height=34, placeholder_text="Name, student no., or PC")
        search_entry.pack(side="left", padx=4)
        ctk.CTkLabel(filter_bar, text="Section", text_color=colors["text_primary"]).pack(side="left", padx=(10, 4))
        section_entry = ctk.CTkEntry(filter_bar, textvariable=section_var, width=140, height=34, placeholder_text="e.g. BSIT 2A")
        section_entry.pack(side="left", padx=4)
        ctk.CTkLabel(filter_bar, text="Date (YYYY-MM-DD)", text_color=colors["text_primary"]).pack(side="left", padx=(10, 4))
        date_entry = ctk.CTkEntry(filter_bar, textvariable=date_var, width=140, height=34, placeholder_text="YYYY-MM-DD")
        date_entry.pack(side="left", padx=4)

        pager = ctk.CTkFrame(filter_bar, fg_color="transparent")
        pager.pack(side="right", padx=10)
        prev_btn = ctk.CTkButton(pager, text="Prev", width=72, height=34, **BUTTON_NEUTRAL)
        prev_btn.pack(side="left", padx=4)
        page_label.pack(side="left", padx=4)
        next_btn = ctk.CTkButton(pager, text="Next", width=72, height=34, **BUTTON_NEUTRAL)
        next_btn.pack(side="left", padx=4)

        body = ctk.CTkScrollableFrame(win, fg_color=colors["card_bg"], border_width=1, border_color=colors["border"])
        body.pack(fill="both", expand=True, padx=10, pady=(0, 6))

        footer = ctk.CTkFrame(win, fg_color="transparent")
        footer.pack(fill="x", padx=12, pady=(0, 12))
        ctk.CTkButton(footer, text="Close", command=win.destroy, width=100, height=34, **BUTTON_NEUTRAL).pack(side="right", padx=4)

        def _pretty_day(day_key: str) -> str:
            try:
                return time.strftime("%B %d, %Y", time.strptime(day_key, "%Y-%m-%d"))
            except Exception:
                return day_key or "Unknown Day"

        def _duration_text(login_ts: Optional[float], logout_ts: Optional[float]) -> str:
            if login_ts is None:
                return "--"
            end_ts = float(logout_ts) if isinstance(logout_ts, (int, float)) else time.time()
            total = max(0, int(end_ts - float(login_ts)))
            hrs = total // 3600
            mins = (total % 3600) // 60
            secs = total % 60
            return f"{hrs:02d}:{mins:02d}:{secs:02d}"

        def _export_rows(rows: list[dict]) -> list[dict]:
            payload: list[dict] = []
            for row in rows:
                pc_id = str(row.get("pc_id", "") or (self.selected_pc if not include_pc else ""))
                login_ts = row.get("login_ts")
                logout_ts = row.get("logout_ts")
                payload.append({
                    "pc_id": pc_id,
                    "student_name": str(row.get("full_name", "")),
                    "student_number": str(row.get("student_number", "")),
                    "login_time": self._format_ts(login_ts),
                    "logout_time": self._format_ts(logout_ts),
                    "session_duration": _duration_text(login_ts, logout_ts),
                })
            return payload

        def export_csv() -> None:
            rows = _export_rows(current_visible_rows)
            if not rows:
                messagebox.showinfo("Export", "No rows available to export.")
                return
            out_dir = ROOT / "exports"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"history_export_{int(time.time())}.csv"
            with out_file.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["PC ID", "Student Name", "Student Number", "Login Time", "Logout Time", "Session Duration"])
                for row in rows:
                    writer.writerow([
                        row["pc_id"],
                        row["student_name"],
                        row["student_number"],
                        row["login_time"],
                        row["logout_time"],
                        row["session_duration"],
                    ])
            messagebox.showinfo("Export", f"Exported {len(rows)} row(s) to: {out_file}")

        def export_pdf() -> None:
            rows = _export_rows(current_visible_rows)
            if not rows:
                messagebox.showinfo("Export", "No rows available to export.")
                return
            out_dir = ROOT / "exports"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"history_export_{int(time.time())}.pdf"

            def _fit(text: str, width: int) -> str:
                value = str(text)
                if len(value) <= width:
                    return value.ljust(width)
                return (value[: max(0, width - 3)] + "...")

            header_line = (
                f"{_fit('PC ID', 8)} | "
                f"{_fit('Student Name', 24)} | "
                f"{_fit('Student Number', 14)} | "
                f"{_fit('Login Time', 19)} | "
                f"{_fit('Logout Time', 19)} | "
                f"{_fit('Session Duration', 14)}"
            )
            lines = [
                title_var.get(),
                f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                header_line,
                "-" * len(header_line),
            ]
            for row in rows:
                lines.append(
                    f"{_fit(row['pc_id'], 8)} | "
                    f"{_fit(row['student_name'], 24)} | "
                    f"{_fit(row['student_number'], 14)} | "
                    f"{_fit(row['login_time'], 19)} | "
                    f"{_fit(row['logout_time'], 19)} | "
                    f"{_fit(row['session_duration'], 14)}"
                )

            font = ImageFont.load_default()
            page_w, page_h = 1654, 2339
            margin = 60
            line_h = 26
            max_lines = max(1, (page_h - (margin * 2)) // line_h)
            pages: list[Image.Image] = []
            cursor = 0
            while cursor < len(lines):
                img = Image.new("RGB", (page_w, page_h), (255, 255, 255))
                draw = ImageDraw.Draw(img)
                y = margin
                for line in lines[cursor: cursor + max_lines]:
                    draw.text((margin, y), str(line)[:190], fill=(0, 0, 0), font=font)
                    y += line_h
                pages.append(img)
                cursor += max_lines

            pages[0].save(out_file, "PDF", resolution=100.0, save_all=True, append_images=pages[1:])
            messagebox.showinfo("Export", f"Exported {len(rows)} row(s) to: {out_file}")

        ctk.CTkButton(footer, text="Export CSV", command=export_csv, width=116, height=34, **BUTTON_NEUTRAL).pack(side="right", padx=4)
        ctk.CTkButton(footer, text="Export PDF", command=export_pdf, width=116, height=34, **BUTTON_NEUTRAL).pack(side="right", padx=4)

        columns = ["Login Time", "Logout Time", "Full Name", "Student Number", "Year/Section", "Status", "Recording"]
        if include_pc:
            columns = ["PC"] + columns

        col_alignments = {
            "PC": "w",
            "Login Time": "e",
            "Logout Time": "e",
            "Full Name": "w",
            "Student Number": "e",
            "Year/Section": "w",
            "Status": "w",
            "Recording": "w",
        }
        column_weights = {
            "PC": 1,
            "Login Time": 2,
            "Logout Time": 2,
            "Full Name": 2,
            "Student Number": 1,
            "Year/Section": 1,
            "Status": 1,
            "Recording": 2,
        }

        def apply_filters() -> None:
            nonlocal filtered_records, current_page, day_keys, grouped_by_day
            q = search_var.get().strip().lower()
            sec = section_var.get().strip().lower()
            dt = date_var.get().strip()
            data = source_records
            if q:
                data = [r for r in data if q in str(r.get("full_name", "")).lower() or q in str(r.get("student_number", "")).lower() or q in str(r.get("pc_id", "")).lower()]
            if sec:
                data = [r for r in data if sec in str(r.get("year_section", "")).lower()]
            if dt:
                data = [r for r in data if self._format_ts(r.get("login_ts")).startswith(dt)]
            filtered_records = data
            grouped: dict[str, list[dict]] = {}
            for r in filtered_records:
                key = self._format_ts(r.get("login_ts"))[:10]
                grouped.setdefault(key, []).append(r)
            day_keys = sorted(grouped.keys(), reverse=True)
            grouped_by_day = grouped
            current_page = 0
            render_table()

        def render_table() -> None:
            nonlocal current_page
            for child in body.winfo_children():
                child.destroy()
            header_row = ctk.CTkFrame(body, fg_color="transparent")
            header_row.pack(fill="x", padx=8, pady=(8, 4))
            for idx, col in enumerate(columns):
                sticky = col_alignments.get(col, "w")
                ctk.CTkLabel(
                    header_row,
                    text=col,
                    font=(self.FONT_FAMILY, 12, "bold"),
                    text_color=colors["text_primary"],
                ).grid(row=0, column=idx, padx=10, pady=(4, 6), sticky=sticky)
                header_row.grid_columnconfigure(idx, weight=column_weights.get(col, 1), uniform="history_cols")

            if not filtered_records:
                title_var.set(title)
                summary_var.set("No matching session records. Adjust the filters or use YYYY-MM-DD for the date field.")
                empty_state = ctk.CTkFrame(body, fg_color=colors["root_bg"], corner_radius=8, border_width=1, border_color=colors["border"])
                empty_state.pack(fill="x", padx=8, pady=(6, 8))
                ctk.CTkLabel(
                    empty_state,
                    text="No session records found.",
                    font=(self.FONT_FAMILY, 12, "bold"),
                    text_color=colors["text_primary"],
                ).pack(anchor="w", padx=12, pady=(12, 2))
                ctk.CTkLabel(
                    empty_state,
                    text="Try a different student name, section, or date filter.",
                    font=(self.FONT_FAMILY, 11),
                    text_color=colors["text_secondary"],
                ).pack(anchor="w", padx=12, pady=(0, 12))
                page_label.configure(text="Page 1/1")
                prev_btn.configure(state="disabled")
                next_btn.configure(state="disabled")
                return

            total_pages = max(1, len(day_keys))
            if current_page >= total_pages:
                current_page = max(0, total_pages - 1)
            active_day = day_keys[current_page] if day_keys else ""
            page_rows = list(grouped_by_day.get(active_day, []))
            title_var.set(f"{title} - {_pretty_day(active_day)}" if active_day else title)
            summary_var.set(
                f"{len(filtered_records)} matching record(s) across {total_pages} day(s). "
                f"Showing {len(page_rows)} row(s) for {_pretty_day(active_day) if active_day else 'the selected range'}."
            )
            page_label.configure(text=f"Page {current_page + 1}/{total_pages}")
            prev_btn.configure(state="normal" if current_page > 0 else "disabled")
            next_btn.configure(state="normal" if current_page + 1 < total_pages else "disabled")

            current_visible_rows.clear()
            current_visible_rows.extend(page_rows)
            for row in page_rows:
                row_card = ctk.CTkFrame(
                    body,
                    fg_color=colors["root_bg"],
                    corner_radius=8,
                    border_width=1,
                    border_color=colors["border"],
                )
                row_card.pack(fill="x", padx=8, pady=4)
                for idx, col in enumerate(columns):
                    row_card.grid_columnconfigure(idx, weight=column_weights.get(col, 1), uniform="history_cols")

                rec_path = str(row.get("recording_path", "") or "")
                values = [
                    self._format_ts(row.get("login_ts")),
                    self._format_ts(row.get("logout_ts")),
                    str(row.get("full_name", "")),
                    str(row.get("student_number", "")),
                    str(row.get("year_section", "")),
                    str(row.get("status", "")),
                    str(Path(rec_path).name) if rec_path else "--",
                ]
                value_columns = ["Login Time", "Logout Time", "Full Name", "Student Number", "Year/Section", "Status", "Recording"]
                col_offset = 0
                if include_pc:
                    pc = str(row.get("pc_id", ""))
                    pc_btn = ctk.CTkButton(
                        row_card,
                        text=pc,
                        width=84,
                        height=30,
                        command=lambda pid=pc, w=win: (setattr(self, "selected_pc", pid), self.selected_history_button.configure(state="normal"), self._open_selected_history(), w.destroy()),
                        **BUTTON_NEUTRAL,
                    )
                    pc_btn.grid(row=0, column=0, padx=10, pady=8, sticky="w")
                    col_offset = 1
                for col_idx, val in enumerate(values):
                    col_name = value_columns[col_idx]
                    sticky = col_alignments.get(col_name, "w")
                    text_color = colors["text_secondary"]
                    if col_name == "Status":
                        text_color = self.STATUS_COLORS.get(val, colors["text_secondary"])
                    target_column = col_idx + col_offset
                    if col_name == "Recording" and rec_path:
                        recording_cell = ctk.CTkFrame(row_card, fg_color="transparent")
                        recording_cell.grid(row=0, column=target_column, padx=10, pady=8, sticky="ew")
                        ctk.CTkLabel(
                            recording_cell,
                            text=val,
                            font=(self.FONT_FAMILY, 12),
                            text_color=colors["text_secondary"],
                            anchor="w",
                        ).pack(side="left", padx=(0, 8))
                        ctk.CTkButton(
                            recording_cell,
                            text="Play",
                            width=64,
                            height=30,
                            command=lambda p=rec_path: self._play_recording(p),
                            **BUTTON_NEUTRAL,
                        ).pack(side="right")
                    else:
                        ctk.CTkLabel(
                            row_card,
                            text=val,
                            font=(self.FONT_FAMILY, 12),
                            text_color=text_color,
                        ).grid(row=0, column=target_column, padx=10, pady=8, sticky=sticky)

        def prev_page() -> None:
            nonlocal current_page
            if current_page > 0:
                current_page -= 1
                render_table()

        def next_page() -> None:
            nonlocal current_page
            total_pages = max(1, len(day_keys))
            if current_page + 1 < total_pages:
                current_page += 1
                render_table()

        prev_btn.configure(command=prev_page)
        next_btn.configure(command=next_page)
        search_entry.bind("<Return>", lambda _e: apply_filters())
        section_entry.bind("<Return>", lambda _e: apply_filters())
        date_entry.bind("<Return>", lambda _e: apply_filters())
        ctk.CTkButton(filter_bar, text="Apply", command=apply_filters, width=88, height=34, **BUTTON_PRIMARY).pack(side="left", padx=8)
        ctk.CTkButton(filter_bar, text="Reset", command=lambda: (search_var.set(""), section_var.set(""), date_var.set(""), apply_filters()), width=88, height=34, **BUTTON_NEUTRAL).pack(side="left", padx=4)
        apply_filters()
        search_entry.focus_set()
    def _open_overall_history(self) -> None:
        records = self.server.get_all_sessions(limit=2000)
        summary: dict[str, dict] = {}
        for row in records:
            pc_id = str(row.get("pc_id", "")).strip()
            if not pc_id:
                continue
            item = summary.setdefault(pc_id, {"count": 0, "last_login_ts": 0.0})
            item["count"] = int(item["count"]) + 1
            login_ts = float(row.get("login_ts") or 0.0)
            if login_ts > float(item["last_login_ts"]):
                item["last_login_ts"] = login_ts

        win = ctk.CTkToplevel(self.root)
        win.title("Overall Session History")
        win.geometry("560x620")
        win.transient(self.root)
        win.bind("<Escape>", lambda _e: win.destroy())
        colors = self._apply_theme_to_toplevel(win)

        header = ctk.CTkFrame(win, fg_color=colors["card_bg"], border_width=1, border_color=colors["border"])
        header.pack(fill="x", padx=10, pady=(10, 8))
        ctk.CTkLabel(header, text="Overall Session History", font=(self.FONT_FAMILY, 16, "bold"), text_color=colors["text_primary"]).pack(anchor="w", padx=12, pady=(10, 4))
        ctk.CTkLabel(header, text="Select a PC to open its session history.", font=(self.FONT_FAMILY, 12), text_color=colors["text_secondary"]).pack(anchor="w", padx=12, pady=(0, 10))

        body = ctk.CTkScrollableFrame(win, fg_color=colors["card_bg"], border_width=1, border_color=colors["border"])
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        footer = ctk.CTkFrame(win, fg_color="transparent")
        footer.pack(fill="x", padx=12, pady=(0, 12))
        ctk.CTkButton(footer, text="Close", command=win.destroy, width=90, **BUTTON_NEUTRAL).pack(side="right", padx=4)

        if not summary:
            ctk.CTkLabel(body, text="No session records found.", text_color=colors["text_secondary"]).pack(anchor="w", padx=10, pady=10)
            return

        ordered_pc_ids = sorted(summary.keys())
        for pc_id in ordered_pc_ids:
            item = summary[pc_id]
            row = ctk.CTkFrame(body, fg_color=colors["card_bg"], border_width=1, border_color=colors["border"])
            row.pack(fill="x", padx=8, pady=6)

            left = ctk.CTkFrame(row, fg_color="transparent")
            left.pack(side="left", fill="x", expand=True, padx=10, pady=8)
            ctk.CTkLabel(left, text=pc_id, font=(self.FONT_FAMILY, 13, "bold"), text_color=colors["text_primary"]).pack(anchor="w")
            last_login = self._format_ts(item.get("last_login_ts") if float(item.get("last_login_ts", 0.0)) > 0 else None)
            ctk.CTkLabel(left, text=f"Sessions: {int(item.get('count', 0))}   Last Login: {last_login}", font=(self.FONT_FAMILY, 11), text_color=colors["text_secondary"]).pack(anchor="w")

            open_btn = ctk.CTkButton(
                row,
                text="Open",
                width=90,
                command=lambda pid=pc_id, w=win: (setattr(self, "selected_pc", pid), self.selected_history_button.configure(state="normal"), self._open_selected_history(), w.destroy()),
                **BUTTON_NEUTRAL,
            )
            open_btn.pack(side="right", padx=10, pady=10)
    def _open_selected_history(self) -> None:
        if not self.selected_pc:
            return
        records = self.server.get_sessions_for_pc(self.selected_pc, limit=500)
        for row in records:
            row["pc_id"] = self.selected_pc
        self._open_history_modal(f"Session History - {self.selected_pc}", records, include_pc=False)
    def _play_recording(self, path: str) -> None:
        if not Path(path).exists():
            return
        def worker() -> None:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                return
            name = f"Recording: {Path(path).name}"
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                cv2.imshow(name, frame)
                if cv2.waitKey(30) & 0xFF == 27:
                    break
            cap.release()
            cv2.destroyWindow(name)

        threading.Thread(target=worker, daemon=True).start()

    def _manual_record_start(self) -> None:
        if self.server.settings.recording_mode != "manual":
            return
        targets = self._selected_targets()
        if not targets:
            messagebox.showerror("Recording", "Select at least one workstation.")
            return
        with self.server.lock:
            offline = [pc for pc in targets if (pc not in self.server.clients or not self.server.clients[pc].online)]
            not_logged = [pc for pc in targets if (pc in self.server.clients and not self.server.clients[pc].current_user)]
            session_map = {pc_id: self.server.clients[pc_id].auth_session_id if pc_id in self.server.clients else None for pc_id in targets}
        if offline:
            messagebox.showerror("Recording", f"Cannot start recording. Offline workstation(s): {', '.join(offline)}")
            return
        if not_logged:
            messagebox.showerror("Recording", f"Cannot start recording. Not logged in workstation(s): {', '.join(not_logged)}")
            return
        for pc_id, session_id in session_map.items():
            self.server._start_recording_for_session(pc_id, session_id, force=True)

    def _manual_record_stop(self) -> None:
        if self.server.settings.recording_mode != "manual":
            return
        targets = self._selected_targets()
        if not targets:
            messagebox.showerror("Recording", "Select at least one workstation.")
            return
        with self.server.lock:
            ids = [pc for pc in targets if pc in self.server.recordings]
        if not ids:
            messagebox.showerror("Recording", "No active manual recording for selected workstation(s).")
            return
        for pc_id in ids:
            self.server._stop_recording(pc_id, status="manual_stop")

    def _open_settings_modal(self) -> None:
        st = self.server.settings
        win = ctk.CTkToplevel(self.root)
        win.title("Settings")
        win.geometry("760x620")
        win.transient(self.root)
        win.lift()
        win.focus_force()
        win.grab_set()
        win.bind("<Escape>", lambda _e: win.destroy())
        colors = self._apply_theme_to_toplevel(win)

        tab = ctk.CTkTabview(win, fg_color=colors["card_bg"])
        tab.pack(fill="both", expand=True, padx=12, pady=12)
        network_tab = tab.add("Network")
        stream_tab = tab.add("Streaming")
        runtime_tab = tab.add("Runtime")
        recording_tab = tab.add("Recording")
        appearance_tab = tab.add("Appearance")
        session_tab = tab.add("Sessions")

        bind_host_var = ctk.StringVar(value=st.teacher_bind_host)
        connect_host_var = ctk.StringVar(value=st.teacher_connect_host)
        main_var = ctk.StringVar(value=st.main_stream_profile)
        prev_var = ctk.StringVar(value=st.preview_stream_profile)
        rec_var = ctk.StringVar(value=st.recording_mode)
        theme_var = ctk.StringVar(value=st.theme_mode)
        reci_var = ctk.StringVar(value=str(st.reconnect_interval_s))
        hb_var = ctk.StringVar(value=str(st.heartbeat_timeout_s))
        q_var = ctk.StringVar(value=st.frame_queue_policy)
        ret_var = ctk.StringVar(value=str(st.recording_retention_days))
        maxgb_var = ctk.StringVar(value=str(st.recording_max_gb))
        sess_var = ctk.StringVar(value=str(st.session_duration_s))
        day_var = ctk.StringVar(value=str(st.daily_limit_s))
        warn_var = ctk.StringVar(value="1" if st.enable_timer_near_limit_notify else "0")
        pause_var = ctk.StringVar(value="1" if st.enable_timer_pause_on_temp_lock else "0")

        def add_tab_note(parent, text: str) -> None:
            ctk.CTkLabel(
                parent,
                text=text,
                font=(self.FONT_FAMILY, 11),
                text_color=colors["text_secondary"],
                justify="left",
                wraplength=680,
            ).pack(anchor="w", padx=12, pady=(10, 4))

        def _human_duration_text(total_s: int) -> str:
            total = max(0, int(total_s))
            hours = total // 3600
            minutes = (total % 3600) // 60
            seconds = total % 60
            parts: list[str] = []
            if hours:
                parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
            if minutes:
                parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
            if seconds or not parts:
                parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
            return ", ".join(parts)

        def _seconds_equivalent_hint(raw: str, fallback: str) -> str:
            try:
                total = max(0, int(raw))
            except ValueError:
                return fallback
            return f"Current value: {total} seconds = {_human_duration_text(total)}."

        def _retention_days_hint(raw: str) -> str:
            try:
                days = max(1, int(raw))
            except ValueError:
                return "Example: 7 keeps recordings for about one week."
            return f"Current value: keep recordings for about {days} day(s)."

        def _recording_max_hint(raw: str) -> str:
            try:
                limit = max(0.1, float(raw))
            except ValueError:
                return "Example: 5.0 means the folder is trimmed after about 5 GB."
            return f"Current value: keep total recordings under about {limit:.1f} GB."

        add_tab_note(
            network_tab,
            "Network settings control how the Teacher/Admin server listens and what IP address students should use. Restart the admin app after changing the bind address.",
        )

        def add_text_field(parent, label: str, var: ctk.StringVar, help_text: str = "") -> ctk.CTkEntry:
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=(10, 2))
            ctk.CTkLabel(row, text=label, width=200, anchor="w", text_color=colors["text_primary"]).pack(side="left")
            entry = ctk.CTkEntry(row, textvariable=var, height=34)
            entry.pack(side="left", fill="x", expand=True)
            if help_text:
                ctk.CTkLabel(
                    parent,
                    text=help_text,
                    font=(self.FONT_FAMILY, 11),
                    text_color=colors["text_secondary"],
                    justify="left",
                    wraplength=500,
                ).pack(anchor="w", padx=(216, 12), pady=(0, 2))
            return entry

        bind_entry = add_text_field(
            network_tab,
            "Admin Bind IP",
            bind_host_var,
            "Use 0.0.0.0 to listen on all network adapters. Changing this requires restarting the admin app.",
        )
        connect_entry = add_text_field(
            network_tab,
            "Student Connect IP",
            connect_host_var,
            "This is the Teacher/Admin computer IP students should enter in their connecting overlay.",
        )
        network_error = ctk.CTkLabel(network_tab, text="", font=(self.FONT_FAMILY, 11), text_color=ESSU_ERROR)
        network_error.pack(anchor="w", padx=(216, 12), pady=(0, 6))

        add_tab_note(
            stream_tab,
            "Streaming controls the live video quality. Higher main quality looks sharper, while lower preview quality keeps the dashboard lighter.",
        )
        ctk.CTkLabel(stream_tab, text="Main Stream Quality", text_color=colors["text_primary"]).pack(anchor="w", padx=12, pady=(10, 4))
        mrow = ctk.CTkFrame(stream_tab, fg_color="transparent")
        mrow.pack(fill="x", padx=12)
        for v in ("360p", "720p", "1080p"):
            ctk.CTkRadioButton(mrow, text=v, variable=main_var, value=v).pack(side="left", padx=8)
        ctk.CTkLabel(
            stream_tab,
            text="Used for the selected live view and recording quality sent to student clients.",
            font=(self.FONT_FAMILY, 11),
            text_color=colors["text_secondary"],
        ).pack(anchor="w", padx=20, pady=(2, 8))

        ctk.CTkLabel(stream_tab, text="Preview Quality", text_color=colors["text_primary"]).pack(anchor="w", padx=12, pady=(14, 4))
        prow = ctk.CTkFrame(stream_tab, fg_color="transparent")
        prow.pack(fill="x", padx=12)
        for v in ("240p", "360p", "720p"):
            ctk.CTkRadioButton(prow, text=v, variable=prev_var, value=v).pack(side="left", padx=8)
        ctk.CTkLabel(
            stream_tab,
            text="Used only for the small workstation previews on the admin dashboard.",
            font=(self.FONT_FAMILY, 11),
            text_color=colors["text_secondary"],
        ).pack(anchor="w", padx=20, pady=(2, 8))

        numeric_entries: dict[str, ctk.CTkEntry] = {}
        error_labels: dict[str, ctk.CTkLabel] = {}

        def add_numeric_field(
            parent,
            key: str,
            label: str,
            var: ctk.StringVar,
            help_text: str = "",
            live_hint=None,
        ) -> None:
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=(10, 2))
            ctk.CTkLabel(row, text=label, width=200, anchor="w", text_color=colors["text_primary"]).pack(side="left")
            entry = ctk.CTkEntry(row, textvariable=var, height=34)
            entry.pack(side="left", fill="x", expand=True)
            if help_text:
                ctk.CTkLabel(
                    parent,
                    text=help_text,
                    font=(self.FONT_FAMILY, 11),
                    text_color=colors["text_secondary"],
                    justify="left",
                    wraplength=500,
                ).pack(anchor="w", padx=(216, 12), pady=(0, 2))
            if live_hint is not None:
                hint_label = ctk.CTkLabel(
                    parent,
                    text=live_hint(var.get()),
                    font=(self.FONT_FAMILY, 11),
                    text_color=colors["text_secondary"],
                    justify="left",
                    wraplength=500,
                )
                hint_label.pack(anchor="w", padx=(216, 12), pady=(0, 2))

                def _refresh_hint(*_args) -> None:
                    hint_label.configure(text=live_hint(var.get()))

                var.trace_add("write", _refresh_hint)
            err_label = ctk.CTkLabel(parent, text="", font=(self.FONT_FAMILY, 11), text_color=ESSU_ERROR)
            err_label.pack(anchor="w", padx=(216, 0), pady=(0, 6))
            numeric_entries[key] = entry
            error_labels[key] = err_label

        add_tab_note(
            runtime_tab,
            "Runtime settings control reconnect timing and client health checks. All time values in this tab are measured in seconds.",
        )
        add_numeric_field(
            runtime_tab,
            "reconnect_interval_s",
            "Reconnect Interval (s)",
            reci_var,
            help_text="How long a student waits before trying to reconnect again when the teacher server is unavailable.",
            live_hint=lambda raw: _seconds_equivalent_hint(raw, "Example: 3 means retry every 3 seconds."),
        )
        add_numeric_field(
            runtime_tab,
            "heartbeat_timeout_s",
            "Heartbeat Timeout (s)",
            hb_var,
            help_text="How long the admin waits without a heartbeat before marking a workstation offline.",
            live_hint=lambda raw: _seconds_equivalent_hint(raw, "Example: 15 means mark the PC offline after 15 seconds with no heartbeat."),
        )

        row = ctk.CTkFrame(runtime_tab, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=10)
        ctk.CTkLabel(row, text="Frame Queue Policy", width=200, anchor="w", text_color=colors["text_primary"]).pack(side="left")
        q_menu = ctk.CTkOptionMenu(row, variable=q_var, values=["freshest", "drop_newest"], height=34)
        q_menu.pack(side="left", fill="x", expand=True)
        queue_note = ctk.CTkLabel(runtime_tab, font=(self.FONT_FAMILY, 11), text_color=colors["text_secondary"], justify="left", wraplength=500)
        queue_note.pack(anchor="w", padx=(216, 12), pady=(0, 6))

        def _refresh_queue_note(*_args) -> None:
            if q_var.get() == "drop_newest":
                queue_note.configure(text="drop_newest ignores new incoming preview frames when the queue is full.")
            else:
                queue_note.configure(text="freshest keeps the latest preview frame by dropping older queued frames when busy.")

        q_var.trace_add("write", _refresh_queue_note)
        _refresh_queue_note()

        add_tab_note(
            recording_tab,
            "Recording settings control automatic capture and cleanup of saved videos on the teacher machine.",
        )
        row = ctk.CTkFrame(recording_tab, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=10)
        ctk.CTkLabel(row, text="Recording Mode", width=200, anchor="w", text_color=colors["text_primary"]).pack(side="left")
        rec_menu = ctk.CTkOptionMenu(row, variable=rec_var, values=["off", "manual", "auto"], height=34)
        rec_menu.pack(side="left", fill="x", expand=True)
        rec_note = ctk.CTkLabel(recording_tab, font=(self.FONT_FAMILY, 11), text_color=colors["text_secondary"], justify="left", wraplength=500)
        rec_note.pack(anchor="w", padx=(216, 12), pady=(0, 6))

        def _refresh_recording_mode_note(*_args) -> None:
            mode = rec_var.get()
            if mode == "manual":
                rec_note.configure(text="manual enables the Start Recording and Stop Recording buttons for selected workstations.")
            elif mode == "auto":
                rec_note.configure(text="auto starts recording automatically when a student session begins.")
            else:
                rec_note.configure(text="off disables automatic recording and manual start/stop actions.")

        rec_var.trace_add("write", _refresh_recording_mode_note)
        _refresh_recording_mode_note()

        add_numeric_field(
            recording_tab,
            "recording_retention_days",
            "Recording Retention (days)",
            ret_var,
            help_text="Recordings older than this many days are deleted automatically.",
            live_hint=_retention_days_hint,
        )
        add_numeric_field(
            recording_tab,
            "recording_max_gb",
            "Recording Max (GB)",
            maxgb_var,
            help_text="When the recordings folder grows past this size, the oldest files are removed first.",
            live_hint=_recording_max_hint,
        )

        rec_actions = ctk.CTkFrame(recording_tab, fg_color="transparent")
        rec_actions.pack(fill="x", padx=12, pady=(10, 6))
        manual_start_btn = ctk.CTkButton(rec_actions, text="Start Recording", command=self._manual_record_start, width=150, height=34, **BUTTON_PRIMARY)
        manual_start_btn.pack(side="left", padx=(0, 6))
        manual_stop_btn = ctk.CTkButton(rec_actions, text="Stop Recording", command=self._manual_record_stop, width=150, height=34, **BUTTON_NEUTRAL)
        manual_stop_btn.pack(side="left", padx=(0, 6))

        def _refresh_manual_record_buttons(*_args) -> None:
            mode = rec_var.get()
            targets = self._selected_targets()
            with self.server.lock:
                valid_start = any((pc in self.server.clients and self.server.clients[pc].online and bool(self.server.clients[pc].current_user)) for pc in targets) if targets else False
                valid_stop = any(pc in self.server.recordings for pc in targets) if targets else False
            if mode != "manual":
                manual_start_btn.configure(state="disabled")
                manual_stop_btn.configure(state="disabled")
            else:
                manual_start_btn.configure(state="normal" if valid_start else "disabled")
                manual_stop_btn.configure(state="normal" if valid_stop else "disabled")

        rec_var.trace_add("write", _refresh_manual_record_buttons)
        _refresh_manual_record_buttons()

        add_tab_note(
            appearance_tab,
            "Appearance changes how the admin dashboard looks. `System` follows the current operating system theme.",
        )
        arow = ctk.CTkFrame(appearance_tab, fg_color="transparent")
        arow.pack(fill="x", padx=12, pady=12)
        ctk.CTkLabel(arow, text="Theme", width=200, anchor="w", text_color=colors["text_primary"]).pack(side="left")
        ctk.CTkOptionMenu(arow, variable=theme_var, values=["light", "dark", "system"], height=34).pack(side="left", fill="x", expand=True)

        add_tab_note(
            session_tab,
            "Session limits are stored in seconds. Examples: 3600 = 1 hour, 5400 = 1 hour 30 minutes, 7200 = 2 hours.",
        )
        ctk.CTkLabel(
            session_tab,
            text="Student messaging and extension requests are core features and stay enabled automatically.",
            font=(self.FONT_FAMILY, 11),
            text_color=colors["text_secondary"],
            justify="left",
            wraplength=680,
        ).pack(anchor="w", padx=12, pady=(0, 6))
        add_numeric_field(
            session_tab,
            "session_duration_s",
            "Session Duration (s)",
            sess_var,
            help_text="Maximum time allowed for one login session before the student must sign in again.",
            live_hint=lambda raw: _seconds_equivalent_hint(raw, "Enter the session duration in seconds. Example: 7200 = 2 hours."),
        )
        add_numeric_field(
            session_tab,
            "daily_limit_s",
            "Daily Limit (s)",
            day_var,
            help_text="Total time a student can use the system in one day across all sessions.",
            live_hint=lambda raw: _seconds_equivalent_hint(raw, "Enter the daily limit in seconds. Example: 7200 = 2 hours."),
        )
        ctk.CTkCheckBox(session_tab, text="Enable Near-Limit Timer Notification", variable=warn_var, onvalue="1", offvalue="0").pack(anchor="w", padx=12, pady=(2, 2))
        ctk.CTkLabel(
            session_tab,
            text="Enables near-limit timer warning events for supported student clients when time is almost over.",
            font=(self.FONT_FAMILY, 11),
            text_color=colors["text_secondary"],
            justify="left",
            wraplength=680,
        ).pack(anchor="w", padx=(36, 12), pady=(0, 4))
        ctk.CTkCheckBox(session_tab, text="Pause Timer on Temporary Lock", variable=pause_var, onvalue="1", offvalue="0").pack(anchor="w", padx=12, pady=(2, 2))
        ctk.CTkLabel(
            session_tab,
            text="Pauses the session countdown during a temporary lock and resumes it after unlock.",
            font=(self.FONT_FAMILY, 11),
            text_color=colors["text_secondary"],
            justify="left",
            wraplength=680,
        ).pack(anchor="w", padx=(36, 12), pady=(0, 8))

        actions = ctk.CTkFrame(win, fg_color="transparent")
        actions.pack(fill="x", padx=12, pady=(0, 12))

        def save_settings() -> None:
            for key, entry in numeric_entries.items():
                entry.configure(border_color=colors["border"])
                error_labels[key].configure(text="")

            parsed: dict[str, float] = {}
            network_error.configure(text="")
            bind_entry.configure(border_color=colors["border"])
            connect_entry.configure(border_color=colors["border"])

            def normalize_host(raw: str, field_name: str) -> str:
                host = str(raw or "").strip()
                if not host:
                    raise ValueError(f"{field_name} is required.")
                if any(ch.isspace() for ch in host):
                    raise ValueError(f"{field_name} cannot contain spaces.")
                if len(host) > 255:
                    raise ValueError(f"{field_name} is too long.")
                return host

            def parse_required_number(key: str, raw: str, caster) -> bool:
                try:
                    parsed[key] = caster(raw)
                    return True
                except ValueError:
                    numeric_entries[key].configure(border_color=ESSU_ERROR)
                    error_labels[key].configure(text="Enter a valid number.")
                    return False

            valid = True
            valid = parse_required_number("reconnect_interval_s", reci_var.get() or "3", int) and valid
            valid = parse_required_number("heartbeat_timeout_s", hb_var.get() or "10", int) and valid
            valid = parse_required_number("recording_retention_days", ret_var.get() or "7", int) and valid
            valid = parse_required_number("recording_max_gb", maxgb_var.get() or "5", float) and valid
            valid = parse_required_number("session_duration_s", sess_var.get() or "7200", int) and valid
            valid = parse_required_number("daily_limit_s", day_var.get() or "7200", int) and valid
            try:
                teacher_bind_host = normalize_host(bind_host_var.get(), "Admin Bind IP")
                teacher_connect_host = normalize_host(connect_host_var.get(), "Student Connect IP")
            except ValueError as exc:
                bind_entry.configure(border_color=ESSU_ERROR)
                connect_entry.configure(border_color=ESSU_ERROR)
                network_error.configure(text=str(exc))
                tab.set("Network")
                return
            if not valid:
                return

            settings = AppSettings(
                teacher_bind_host=teacher_bind_host,
                teacher_connect_host=teacher_connect_host,
                main_stream_profile=main_var.get(),
                preview_stream_profile=prev_var.get(),
                recording_mode=rec_var.get(),
                theme_mode=theme_var.get(),
                reconnect_interval_s=max(1, int(parsed["reconnect_interval_s"])),
                heartbeat_timeout_s=max(2, int(parsed["heartbeat_timeout_s"])),
                frame_queue_policy=q_var.get(),
                recording_retention_days=max(1, int(parsed["recording_retention_days"])),
                recording_max_gb=max(0.1, float(parsed["recording_max_gb"])),
                session_duration_s=max(60, int(parsed["session_duration_s"])),
                daily_limit_s=max(60, int(parsed["daily_limit_s"])),
                enable_session_messaging=True,
                enable_extension_requests=True,
                enable_timer_near_limit_notify=(warn_var.get() == "1"),
                enable_timer_pause_on_temp_lock=(pause_var.get() == "1"),
            )
            self.server.update_settings(settings)
            mode = settings.theme_mode if settings.theme_mode in {"light", "dark", "system"} else "system"
            ctk.set_appearance_mode(mode)
            self._apply_theme_to_ui()
            win.destroy()

        ctk.CTkButton(actions, text="Overall Session History", command=self._open_overall_history, height=34, **BUTTON_NEUTRAL).pack(side="left", padx=4)
        # ctk.CTkButton(actions, text="Students", command=self._open_student_management, **BUTTON_NEUTRAL).pack(side="left", padx=4)
        # ctk.CTkButton(actions, text="Reservations", command=self._open_reservation_management, **BUTTON_NEUTRAL).pack(side="left", padx=4)
        ctk.CTkButton(actions, text="Close", command=win.destroy, height=34, **BUTTON_NEUTRAL).pack(side="right", padx=4)
        ctk.CTkButton(actions, text="Save Settings", command=save_settings, height=34, **BUTTON_PRIMARY).pack(side="right", padx=4)

    def _open_student_management(self) -> None:
        self.student_management_panel.open()

    def _open_reservation_management(self) -> None:
        self.reservation_management_panel.open()

    def _update_sensor_panel(self, pc_id: str) -> None:
        sensor = self.server.sensors.get(pc_id, SensorState())
        client = self.server.clients.get(pc_id)
        status, color = self._status_for_pc(pc_id)
        temp = "--" if sensor.temperature is None else f"{sensor.temperature:.2f}°C"
        rpm = "--" if sensor.fan_rpm is None else str(sensor.fan_rpm)
        cpu = "--" if not client or client.cpu_percent is None else f"{client.cpu_percent:.0f}%"
        ram = "--" if not client or client.ram_percent is None else f"{client.ram_percent:.0f}%"
        disk = "--" if not client or client.disk_percent is None else f"{client.disk_percent:.0f}%"
        if not client or client.uptime_s is None:
            uptime = "--"
        else:
            hrs = client.uptime_s // 3600
            mins = (client.uptime_s % 3600) // 60
            secs = client.uptime_s % 60
            uptime = f"{hrs:02d}:{mins:02d}:{secs:02d}"
        user_name = client.current_user if client and client.current_user else "--"
        student_number = client.student_number if client and client.student_number else "--"

        session_left = "--"
        session_left_color = self._theme_palette()["text_primary"]
        if client and client.session_timer:
            rem = max(0, self.server._timer_remaining_ms(client.session_timer, time.time()) // 1000)

            rem_h = rem // 3600
            rem_m = (rem % 3600) // 60
            rem_s = rem % 60
            session_left = f"{rem_h:02d}:{rem_m:02d}:{rem_s:02d}"
            warning_rem_s = max(1, int(self.server.settings.session_duration_s * 0.25))
            critical_rem_s = max(1, int(self.server.settings.session_duration_s * 0.10))
            if rem <= critical_rem_s:
                session_left_color = ESSU_ERROR
            elif rem <= warning_rem_s:
                session_left_color = ESSU_WARNING

        timer_remaining = "--"
        timer_extended = "00:00"
        with self.server.lock:
            now = time.time()
            remaining_values = [
                max(0, int(timer.duration_ms - ((now - timer.start_ts) * 1000)))
                for timer in self.server.timers.values()
                if timer.active and pc_id in timer.targets
            ]
        if remaining_values:
            remaining_ms = min(remaining_values)
            mins, secs = divmod(remaining_ms // 1000, 60)
            timer_remaining = f"{mins:02d}:{secs:02d}"

        ext_ms = int(self.extended_timer_ms_by_pc.get(pc_id, 0))
        ext_mins, ext_secs = divmod(max(0, ext_ms) // 1000, 60)
        timer_extended = f"{ext_mins:02d}:{ext_secs:02d}"

        sensor_age = "--"
        sensor_age_color = self._theme_palette()["text_primary"]
        if sensor.last_update > 0:
            sensor_age_s = int(time.time() - sensor.last_update)
            sensor_age = f"{sensor_age_s}s"
            sensor_warning_s = max(1, int(RUNTIME.sensor_timeout_s * 0.5))
            sensor_critical_s = max(1, int(RUNTIME.sensor_timeout_s))
            if sensor_age_s > sensor_critical_s:
                sensor_age_color = ESSU_ERROR
            elif sensor_age_s > sensor_warning_s:
                sensor_age_color = ESSU_WARNING

        if self.server.health_history:
            latest_health = self.server.health_history[-1]
            system_line = f"System Online: {latest_health.online_clients}/{latest_health.total_clients}"
        else:
            system_line = "System Online: --"

        neutral = self._theme_palette()["text_primary"]
        self.sensor_value_labels["pc"].configure(text=pc_id, text_color=neutral)
        self.sensor_value_labels["temp"].configure(text=temp, text_color=neutral)
        self.sensor_value_labels["rpm"].configure(text=rpm, text_color=neutral)
        self.sensor_value_labels["cpu"].configure(text=cpu, text_color=neutral)
        self.sensor_value_labels["ram"].configure(text=ram, text_color=neutral)
        self.sensor_value_labels["disk"].configure(text=disk, text_color=neutral)
        self.sensor_value_labels["uptime"].configure(text=uptime, text_color=neutral)
        self.sensor_value_labels["user"].configure(text=user_name, text_color=neutral)
        self.sensor_value_labels["student_number"].configure(text=student_number, text_color=neutral)
        self.sensor_value_labels["session_left"].configure(text=session_left, text_color=session_left_color)
        self.sensor_value_labels["status"].configure(text=self._display_status(status), text_color=color)
        self.sensor_value_labels["timer_extended"].configure(text=timer_extended, text_color=neutral)
        self.sensor_value_labels["sensor_age"].configure(text=sensor_age, text_color=sensor_age_color)
        self.sensor_value_labels["system"].configure(text=system_line.replace("System Online: ", ""), text_color=neutral)


    def _update_tile_frame(self, pc_id: str, image: Optional[Image.Image]) -> None:
        now_ts = time.time()
        last_ts = float(self._last_tile_repaint_ts.get(pc_id, 0.0))
        if (now_ts - last_ts) < self.MIN_TILE_REPAINT_S:
            return
        self._last_tile_repaint_ts[pc_id] = now_ts
        self._ensure_tile(pc_id)
        client = self.server.clients.get(pc_id)
        online = bool(client and client.online)
        base_image = image if online and image is not None else self._render_offline_tile(pc_id)
        preview_image = self._apply_preview_overlay(base_image, pc_id) if online else base_image

        thumb = preview_image.copy()
        thumb.thumbnail((460, 300))
        if self.server.settings.preview_stream_profile == "240p":
            thumb = thumb.resize((320, 180)).resize((460, 300))
        elif self.server.settings.preview_stream_profile == "360p":
            thumb = thumb.resize((640, 360)).resize((460, 300))
        elif self.server.settings.preview_stream_profile == "720p":
            thumb = thumb.resize((1280, 720)).resize((460, 300))
        status, color = self._status_for_pc(pc_id)
        thumb = self._apply_preview_status_dot(thumb, color)
        photo = ImageTk.PhotoImage(thumb)
        preview = self.tiles[pc_id]["preview"]
        tile = self.tiles[pc_id]["tile"]
        assert isinstance(preview, ctk.CTkLabel)
        assert isinstance(tile, ctk.CTkFrame)
        is_selected = self.selected_pc == pc_id
        self._configure_if_changed(
            tile,
            fg_color="transparent",
            border_color=self.SELECTED_BORDER if is_selected else self.TILE_BORDER,
            border_width=2 if is_selected else 1,
        )
        indicator = self.tiles[pc_id]["indicator"]
        assert isinstance(indicator, ctk.CTkLabel)
        self._configure_if_changed(indicator, text="", fg_color="transparent", text_color=color)
        self._configure_if_changed(preview, text="", fg_color="transparent")
        preview.configure(image=photo)
        preview.image = photo

        if self.selected_pc == pc_id:
            large_now_ts = time.time()
            last_large_ts = float(self._last_large_repaint_ts.get(pc_id, 0.0))
            if (large_now_ts - last_large_ts) < self.MIN_TILE_REPAINT_S:
                return
            self._last_large_repaint_ts[pc_id] = large_now_ts
            if online:
                # Fit the image to the fixed preview host so it never reflows the dashboard layout.
                width = self.large_view_host.winfo_width() if hasattr(self, "large_view_host") else self.large_view.winfo_width()
                height = self.large_view_host.winfo_height() if hasattr(self, "large_view_host") else self.large_view.winfo_height()

                if width > 1 and height > 1:
                    display_img = self._apply_main_overlay(base_image, pc_id)
                    target_w, target_h = 1280, 720
                    fit_w = min(width, target_w)
                    fit_h = min(height, target_h)
                    display_img.thumbnail((fit_w, fit_h))
                    large_photo = ImageTk.PhotoImage(display_img)
                    try:
                        self._configure_if_changed(self.large_view, text="", compound="top", font=("Arial", 24, "bold"), fg_color="transparent")
                        self.large_view.configure(image=large_photo)
                    except tk.TclError:
                        self._recreate_large_view()
                        self._configure_if_changed(self.large_view, text="", compound="top", font=("Arial", 24, "bold"), fg_color="transparent")
                        self.large_view.configure(image=large_photo)
                    self.large_view.image = large_photo
            else:
                try:
                    self._configure_if_changed(self.large_view, image=None, text="Selected workstation is offline.", text_color=self._theme_palette()["text_secondary"], fg_color="transparent")
                except tk.TclError:
                    self._recreate_large_view()
                    self._configure_if_changed(self.large_view, image=None, text="Selected workstation is offline.", text_color=self._theme_palette()["text_secondary"], fg_color="transparent")
                self.large_view.image = None
            self._update_sensor_panel(pc_id)

    def _recreate_large_view(self) -> None:
        try:
            self.large_view.destroy()
        except Exception:
            pass
        colors = self._theme_palette()
        self.large_view = ctk.CTkLabel(
            self.large_view_host,
            text="Select a workstation to view its live feed.",
            anchor="center",
            justify="center",
            wraplength=520,
            font=(self.FONT_FAMILY, 18, "bold"),
            fg_color="transparent",
            text_color=colors["text_secondary"],
            corner_radius=8,
        )
        self.large_view.pack(fill="both", expand=True)

    def _drain_queues(self) -> None:
        try:
            self._refresh_runtime_notice()

            # Update changed thumbnails only
            changed: set[str] = set(self._pending_pc_updates)
            self._pending_pc_updates.clear()
            latest_frames: dict[str, Image.Image] = {}
            try:
                while True:
                    pc_id, image = self.server.frame_queue.get_nowait()
                    latest_frames[pc_id] = image
                    changed.add(pc_id)
            except queue.Empty:
                pass
            if latest_frames:
                with self.server.lock:
                    for pc_id, image in latest_frames.items():
                        if pc_id in self.server.clients:
                            self.server.clients[pc_id].last_frame = image

            try:
                while True:
                    changed.add(self.server.status_queue.get_nowait())
            except queue.Empty:
                pass

            try:
                while True:
                    changed.add(self.server.sensor_queue.get_nowait())
            except queue.Empty:
                pass

            ordered_changed = sorted(changed)
            if ordered_changed:
                start_idx = self._drain_rr_index % len(ordered_changed)
                rotated_changed = ordered_changed[start_idx:] + ordered_changed[:start_idx]
                processed = 0
                for idx, pc_id in enumerate(rotated_changed):
                    if idx >= self.MAX_TILE_UPDATES_PER_DRAIN:
                        self._pending_pc_updates.update(rotated_changed[idx:])
                        break
                    image = latest_frames.get(pc_id)
                    if image is None:
                        with self.server.lock:
                            image = self.server.clients[pc_id].last_frame if pc_id in self.server.clients else None
                    try:
                        self._update_tile_frame(pc_id, image)
                    except tk.TclError:
                        self._recreate_large_view()
                    processed += 1
                self._drain_rr_index = (start_idx + processed) % len(ordered_changed)

            if self.selected_pc:
                self._update_sensor_panel(self.selected_pc)

            self._refresh_control_buttons()
        except Exception as exc:
            self.server._log_event("ui_drain_error", reason=str(exc))
            self._set_runtime_notice("Dashboard recovered after a refresh issue.", ESSU_WARNING, hold_s=20.0)
        finally:
            try:
                self.root.after(100, self._drain_queues)
            except tk.TclError:
                pass

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    server = TeacherDeployServer()
    server.start()
    TeacherDeployUI(server).run()


if __name__ == "__main__":
    main()
