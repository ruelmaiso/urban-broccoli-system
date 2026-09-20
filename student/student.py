import json
import logging
import queue
import random
import socket
import threading
import time
import traceback
import uuid
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Callable, Optional
from logging.handlers import RotatingFileHandler

import cv2
import customtkinter as ctk
import mss
import numpy
import os
import platform
import psutil
try:
    import sounddevice as sd
except Exception:
    sd = None
from PIL import Image, ImageTk
import sys

# ---- PyInstaller-safe resource path ----
def resource_path(relative_path: str) -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / relative_path
    return Path(__file__).resolve().parents[1] / relative_path


# ---- Project root (for imports only) ----
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.deploy_settings import NETWORK, RUNTIME
from core.protocol import recv_frame, recv_json_line, send_frame, send_json
from core.student_settings import StudentSettings, StudentSettingsStore, normalize_teacher_host

STUDENT_LOG_DIR = ROOT / "logs"
STUDENT_LOG_FILE = STUDENT_LOG_DIR / "student_runtime.log"


def _student_logger() -> logging.Logger:
    STUDENT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("student-runtime")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = RotatingFileHandler(STUDENT_LOG_FILE, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger

AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_DTYPE = "int16"
AUDIO_BLOCK_MS = 40
AUDIO_BLOCK_SIZE = max(1, int(AUDIO_SAMPLE_RATE * AUDIO_BLOCK_MS / 1000))


DEV_MODE = True


class OverlayState(Enum):
    HIDDEN = "hidden"
    AUTH_REQUIRED = "auth_required"
    LOCKED_TEMPORARY = "locked_temporary"
    CONNECTING = "connecting"
    BROADCAST = "broadcast"


class OverlayController:
    def __init__(self) -> None:
        ctk.set_appearance_mode("dark")
        self._root: Optional[ctk.CTk] = None
        self._lock_frame: Optional[ctk.CTkFrame] = None
        self._auth_frame: Optional[ctk.CTkFrame] = None
        self._connecting_frame: Optional[ctk.CTkFrame] = None
        self._broadcast_frame: Optional[ctk.CTkFrame] = None
        self._broadcast_image_label: Optional[ctk.CTkLabel] = None
        self._broadcast_status_label: Optional[ctk.CTkLabel] = None
        self._broadcast_photo: Optional[ImageTk.PhotoImage] = None
        # This is deliberately independent of the Tk command queue.  A receiver
        # thread can invalidate a session while an old frame is already queued.
        self._broadcast_generation = 0
        self._broadcast_generation_lock = threading.Lock()
        self._state = OverlayState.HIDDEN
        self._cmd_q: "queue.Queue[tuple[str, dict, threading.Event, dict]]" = queue.Queue()
        self._worker_started = False
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_lock = threading.Lock()
        self._auth_result_q: "queue.Queue[Optional[dict]]" = queue.Queue()
        self._auth_handlers: Optional[tuple] = None
        self._teacher_host_getter: Optional[Callable[[], str]] = None
        self._teacher_host_saver: Optional[Callable[[str], tuple[bool, str]]] = None

      # NEW: deterministic UI bootstrap synchronization
        self._ui_ready_event = threading.Event()

        self._logo_image: Optional[ctk.CTkImage] = None
        self._auth_logo_label: Optional[ctk.CTkLabel] = None
        self._toast_label: Optional[ctk.CTkLabel] = None
        self._toast_after_id = None
        self._chat_send_handler: Optional[Callable[[str], bool]] = None
        self._chat_history: list[dict] = []
        self._chat_window: Optional[ctk.CTkToplevel] = None
        self._chat_enabled = False
        self._chat_launcher: Optional[ctk.CTkToplevel] = None
        self._chat_launcher_button: Optional[ctk.CTkButton] = None
        self._chat_timer_provider: Optional[Callable[[], Optional[int]]] = None
        self._chat_timer_after_id = None
        self._chat_launcher_pos: Optional[tuple[int, int]] = None
        self._chat_launcher_drag_state: Optional[dict] = None
        self._chat_launcher_suppress_click_until = 0.0
        self._broadcast_diag_callback: Optional[Callable[[str, dict], None]] = None

        logo_path = resource_path("assets/essu_logo.png")
        try:
            if logo_path.exists():
                logo = Image.open(logo_path)
                self._logo_image = ctk.CTkImage(light_image=logo, dark_image=logo, size=(110, 110))
        except Exception:
            self._logo_image = None

    def _ensure_worker(self) -> None:
        with self._worker_lock:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                return
            self._worker_started = True
            self._ui_ready_event.clear()
            self._worker_thread = threading.Thread(target=self._ui_loop, daemon=True)
            self._worker_thread.start()

    def set_broadcast_diag_callback(self, callback: Callable[[str, dict], None]) -> None:
        self._broadcast_diag_callback = callback

    def _emit_broadcast_diag(self, event: str, **data) -> None:
        if self._broadcast_diag_callback is not None:
            self._broadcast_diag_callback(event, data)

    def _build_connecting_frame(self) -> ctk.CTkFrame:
        assert self._root is not None

        frame = ctk.CTkFrame(self._root, fg_color="#0F172A")
        frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        card = ctk.CTkFrame(
            frame,
            width=560,
            fg_color="#1E293B",
            corner_radius=20,
            border_width=1,
            border_color="#334155",
        )
        card.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(
            card,
            text="Connecting to Server",
            font=("Arial", 28, "bold"),
            text_color="#F8FAFC",
        ).pack(pady=(36, 12), padx=32)

        ctk.CTkLabel(
            card,
            text="This workstation will continue automatically when the server becomes available.",
            font=("Arial", 14),
            text_color="#94A3B8",
            justify="center",
            wraplength=430,
        ).pack(pady=(0, 10), padx=32)

        ctk.CTkLabel(
            card,
            text="Retrying connection.",
            font=("Arial", 13, "bold"),
            text_color="#60A5FA",
        ).pack(pady=(0, 14))

        settings_box = ctk.CTkFrame(card, fg_color="#0F172A", corner_radius=14)
        settings_box.pack(fill="x", padx=32, pady=(0, 34))
        ctk.CTkLabel(
            settings_box,
            text="Teacher/Admin Server IP",
            font=("Arial", 13, "bold"),
            text_color="#E2E8F0",
        ).pack(anchor="w", padx=16, pady=(14, 4))

        current_host = ""
        if self._teacher_host_getter is not None:
            try:
                current_host = self._teacher_host_getter()
            except Exception:
                current_host = ""
        host_var = ctk.StringVar(value=current_host)
        host_entry = ctk.CTkEntry(
            settings_box,
            textvariable=host_var,
            placeholder_text="Example: 192.168.1.157",
            height=34,
        )
        host_entry.pack(fill="x", padx=16, pady=(0, 8))
        status_label = ctk.CTkLabel(
            settings_box,
            text="Change this if the teacher/admin computer uses a different IP.",
            font=("Arial", 11),
            text_color="#94A3B8",
            wraplength=460,
            justify="left",
        )
        status_label.pack(anchor="w", padx=16, pady=(0, 10))

        def save_host() -> None:
            if self._teacher_host_saver is None:
                status_label.configure(text="Server IP settings are unavailable.", text_color="#EF4444")
                return
            ok, message = self._teacher_host_saver(host_var.get())
            status_label.configure(text=message, text_color="#22C55E" if ok else "#EF4444")

        ctk.CTkButton(
            settings_box,
            text="Save & Reconnect",
            command=save_host,
            height=32,
            fg_color="#2563EB",
            hover_color="#1D4ED8",
        ).pack(anchor="e", padx=16, pady=(0, 14))

        return frame

    def _build_broadcast_frame(self) -> ctk.CTkFrame:
        assert self._root is not None
        frame = ctk.CTkFrame(self._root, fg_color="#020617")
        frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        self._broadcast_image_label = ctk.CTkLabel(
            frame,
            text="Screen share is starting...",
            text_color="#E2E8F0",
            font=("Arial", 24, "bold"),
            fg_color="transparent",
            anchor="center",
            justify="center",
        )
        self._broadcast_image_label.pack(fill="both", expand=True, padx=18, pady=(18, 8))

        self._broadcast_status_label = ctk.CTkLabel(
            frame,
            text="Admin Screen Share",
            text_color="#94A3B8",
            font=("Arial", 13),
            fg_color="transparent",
        )
        self._broadcast_status_label.pack(pady=(0, 18))
        return frame

    def _clear_broadcast_ui(self) -> None:
        if self._broadcast_image_label is not None and self._broadcast_image_label.winfo_exists():
            self._broadcast_image_label.configure(
                image=None,
                text="Admin screen share is starting...",
                compound="center",
            )
            self._broadcast_image_label.image = None
        self._broadcast_photo = None

    def _hide_broadcast(self) -> None:
        if self._broadcast_frame is not None and self._broadcast_frame.winfo_exists():
            self._broadcast_frame.place_forget()

    def _show_broadcast(self) -> None:
        assert self._root is not None
        if self._auth_frame is not None and self._auth_frame.winfo_exists():
            self._auth_frame.place_forget()
        if self._lock_frame is not None and self._lock_frame.winfo_exists():
            self._lock_frame.place_forget()
        if self._connecting_frame is not None and self._connecting_frame.winfo_exists():
            self._connecting_frame.place_forget()
        if self._broadcast_frame is None or not self._broadcast_frame.winfo_exists():
            self._broadcast_frame = self._build_broadcast_frame()
        else:
            self._broadcast_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._broadcast_frame.lift()

    def _render_broadcast_frame(self, image: Optional[Image.Image]) -> None:
        assert self._root is not None
        if image is None:
            self._clear_broadcast_ui()
            return
        self._emit_broadcast_diag("ui_render_enter")
        if self._broadcast_frame is None or not self._broadcast_frame.winfo_exists():
            self._broadcast_frame = self._build_broadcast_frame()
        if self._broadcast_image_label is None or not self._broadcast_image_label.winfo_exists():
            return
        display = image.copy()
        screen_w = max(1, int(self._root.winfo_screenwidth()))
        screen_h = max(1, int(self._root.winfo_screenheight()))
        display.thumbnail((screen_w, screen_h))
        self._broadcast_photo = ImageTk.PhotoImage(display)
        self._broadcast_image_label.configure(image=self._broadcast_photo, text="", compound="center")
        self._broadcast_image_label.image = self._broadcast_photo
        self._emit_broadcast_diag("ui_rendered")
        if self._broadcast_status_label is not None and self._broadcast_status_label.winfo_exists():
            self._broadcast_status_label.configure(text="Live")

    def _build_lock_frame(self) -> ctk.CTkFrame:
        assert self._root is not None

        frame = ctk.CTkFrame(self._root, fg_color="#0F172A")
        frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        screen_w = self._root.winfo_screenwidth()
        screen_h = self._root.winfo_screenheight()
        card_w = min(600, int(screen_w * 0.8))
        card_h = min(240, int(screen_h * 0.6))

        lock_card = ctk.CTkFrame(
            frame,
            width=card_w,
            height=card_h,
            fg_color="#1E293B",
            corner_radius=18,
            border_width=1,
            border_color="#334155",
        )
        lock_card.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(
            lock_card,
            text="Session Paused",
            font=("Arial", 28, "bold"),
            text_color="#F8FAFC",
        ).pack(pady=(42, 12))

        ctk.CTkLabel(
            lock_card,
            text="This PC is temporarily paused.",
            font=("Arial", 14),
            text_color="#94A3B8",
        ).pack(pady=(0, 24))

        return frame
    def _show_lock(self) -> None:
        assert self._root is not None

        if self._auth_frame is not None and self._auth_frame.winfo_exists():
            self._auth_frame.place_forget()
        if self._connecting_frame is not None and self._connecting_frame.winfo_exists():
            self._connecting_frame.place_forget()
        self._hide_broadcast()

        if self._lock_frame is None or not self._lock_frame.winfo_exists():
            self._lock_frame = self._build_lock_frame()
        else:
            self._lock_frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        if self._chat_launcher is not None and self._chat_launcher.winfo_exists():
            self._chat_launcher.lift()
            self._chat_launcher.attributes("-topmost", True)

    def _show_connecting(self) -> None:
        assert self._root is not None

        if self._auth_frame is not None and self._auth_frame.winfo_exists():
            self._auth_frame.place_forget()
        if self._lock_frame is not None and self._lock_frame.winfo_exists():
            self._lock_frame.place_forget()
        self._hide_broadcast()

        if self._connecting_frame is None or not self._connecting_frame.winfo_exists():
            self._connecting_frame = self._build_connecting_frame()
        else:
            self._connecting_frame.place(relx=0, rely=0, relwidth=1, relheight=1)

    def _show_auth(self, send_login, send_register) -> None:
        assert self._root is not None
        if self._lock_frame is not None and self._lock_frame.winfo_exists():
            self._lock_frame.place_forget()
        if self._connecting_frame is not None and self._connecting_frame.winfo_exists():
            self._connecting_frame.place_forget()
        self._hide_broadcast()
        if self._auth_frame is not None and self._auth_frame.winfo_exists():
            self._auth_frame.destroy()

        container = ctk.CTkFrame(self._root, fg_color="#0F172A")
        container.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._auth_frame = container

        card = ctk.CTkFrame(
            container,
            width=500,
            fg_color="#1E293B",
            corner_radius=20,
            border_width=1,
            border_color="#334155",
        )
        card.place(relx=0.5, rely=0.5, anchor="center")

        register_entries: dict[str, ctk.CTkEntry] = {}

        def style_entry(entry: ctk.CTkEntry) -> None:
            entry.configure(
                height=40,
                corner_radius=10,
                border_width=1,
                border_color="#334155",
                fg_color="#0F172A",
                text_color="#F8FAFC",
            )

        def style_primary(btn: ctk.CTkButton) -> None:
            btn.configure(
                height=42,
                corner_radius=10,
                fg_color="#3B82F6",
                hover_color="#2563EB",
                text_color="#F8FAFC",
                font=("Arial", 14, "bold"),
            )

        def style_secondary(btn: ctk.CTkButton) -> None:
            btn.configure(
                height=40,
                corner_radius=10,
                fg_color="#1F2937",
                hover_color="#334155",
                text_color="#94A3B8",
                font=("Arial", 13),
            )

        def clear_card() -> None:
            for widget in card.winfo_children():
                widget.destroy()

        def render_header(title: str) -> None:
            if self._logo_image is not None:
                logo_label = ctk.CTkLabel(card, text="", image=self._logo_image, fg_color="transparent")
                logo_label.pack(pady=(34, 10))
                logo_label.image = self._logo_image
                self._auth_logo_label = logo_label
            ctk.CTkLabel(
                card,
                text=title,
                font=("Arial", 26, "bold"),
                text_color="#F8FAFC",
            ).pack(pady=(0, 22))

        def switch_to_register() -> None:
            clear_card()
            render_header("Student Registration")

            fields = [
                ("full_name", "Full Name"),
                ("year_section", "Year / Section"),
                ("student_number", "Student Number (00-00000)"),
                ("password", "Password"),
            ]
            register_entries.clear()
            for key, label in fields:
                ctk.CTkLabel(
                    card,
                    text=label,
                    font=("Arial", 13),
                    text_color="#94A3B8",
                ).pack(anchor="w", padx=40, pady=(8, 4))
                entry = ctk.CTkEntry(card, width=420, show="*" if key == "password" else None)
                style_entry(entry)
                entry.pack(padx=40, pady=(0, 4))
                register_entries[key] = entry

            status_reg = ctk.CTkLabel(card, text="", font=("Arial", 12), text_color="#EF4444")
            status_reg.pack(pady=(10, 0))

            def on_register() -> None:
                payload = {k: e.get().strip() for k, e in register_entries.items()}
                if not all(payload.values()):
                    status_reg.configure(text="Please complete all fields.", text_color="#EF4444")
                    return
                resp = send_register(payload)
                if resp is None:
                    self._auth_result_q.put(None)
                    return
                if resp.get("ok"):
                    switch_to_login("Registration successful. Please login.", "#10B981")
                    return
                status_reg.configure(text="Registration failed. Please check fields and try again.", text_color="#EF4444")

            register_btn = ctk.CTkButton(card, text="Register", command=on_register, width=420)
            style_primary(register_btn)
            register_btn.pack(pady=(20, 10))

            back_btn = ctk.CTkButton(card, text="Back to Login", command=lambda: switch_to_login("", "#EF4444"), width=420)
            style_secondary(back_btn)
            back_btn.pack(pady=(0, 32))

        def switch_to_login(initial_text: str = "", color: str = "#EF4444") -> None:
            clear_card()
            render_header("Student Login")

            ctk.CTkLabel(card, text="Student Number", font=("Arial", 13), text_color="#94A3B8").pack(anchor="w", padx=40, pady=(8, 4))
            sn = ctk.CTkEntry(card, width=420)
            style_entry(sn)
            sn.pack(padx=40, pady=(0, 8))

            ctk.CTkLabel(card, text="Password", font=("Arial", 13), text_color="#94A3B8").pack(anchor="w", padx=40, pady=(8, 4))
            pw = ctk.CTkEntry(card, width=420, show="*")
            style_entry(pw)
            pw.pack(padx=40, pady=(0, 8))

            status_login = ctk.CTkLabel(card, text=initial_text, font=("Arial", 12), text_color=color)
            status_login.pack(pady=(10, 0))

            def on_login() -> None:
                payload = {"student_number": sn.get().strip(), "password": pw.get().strip()}
                if not payload["student_number"] or not payload["password"]:
                    status_login.configure(text="Student number and password are required.", text_color="#EF4444")
                    return
                resp = send_login(payload)
                if resp is None:
                    self._auth_result_q.put(None)
                    return
                if resp.get("ok") and isinstance(resp.get("user"), dict):
                    self._auth_result_q.put(resp.get("user"))
                    return
                if resp.get("reason") == "reserved_for_another_student":
                    msg = str(resp.get("message") or "This workstation is reserved for another during this time.")
                    status_login.configure(text=msg, text_color="#EF4444")
                    self.notify_message_async(msg)
                    return
                status_login.configure(text="Invalid credentials. Please try again.", text_color="#EF4444")

            login_btn = ctk.CTkButton(card, text="Login", command=on_login, width=420)
            style_primary(login_btn)
            login_btn.pack(pady=(20, 10))

            register_btn = ctk.CTkButton(card, text="Register", command=switch_to_register, width=420)
            style_secondary(register_btn)
            register_btn.pack(pady=(0, 34))

        switch_to_login()

    def set_chat_sender(self, sender: Callable[[str], bool]) -> None:
        self._chat_send_handler = sender

    def set_teacher_host_settings(
        self,
        host_getter: Callable[[], str],
        host_saver: Callable[[str], tuple[bool, str]],
    ) -> None:
        self._teacher_host_getter = host_getter
        self._teacher_host_saver = host_saver


    def set_chat_timer_provider(self, provider: Callable[[], Optional[int]]) -> None:
        self._chat_timer_provider = provider

    def _format_chat_countdown(self, remaining_s: Optional[int]) -> str:
        if remaining_s is None:
            return "--:--:--"
        total = max(0, int(remaining_s))
        hours = total // 3600
        minutes = (total % 3600) // 60
        seconds = total % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _update_chat_launcher_label(self) -> None:
        if self._chat_launcher_button is None:
            return
        remaining_s = self._chat_timer_provider() if self._chat_timer_provider is not None else None
        self._chat_launcher_button.configure(text=f" {self._format_chat_countdown(remaining_s)}")

    def _schedule_chat_launcher_label_update(self) -> None:
        if self._root is None:
            return
        if self._chat_timer_after_id is not None:
            try:
                self._root.after_cancel(self._chat_timer_after_id)
            except Exception:
                pass
            self._chat_timer_after_id = None
        if not self._chat_enabled:
            return

        def _tick() -> None:
            self._chat_timer_after_id = None
            if not self._chat_enabled:
                return
            self._update_chat_launcher_label()
            self._schedule_chat_launcher_label_update()

        self._chat_timer_after_id = self._root.after(1000, _tick)

    def _chat_launcher_metrics(self) -> tuple[int, int, int, int]:
        launcher = self._chat_launcher if self._chat_launcher is not None else self._root
        assert launcher is not None
        screen_w = int(launcher.winfo_screenwidth())
        screen_h = int(launcher.winfo_screenheight())
        width = max(100, min(164, int(screen_w * 0.09)))
        height = 36
        return screen_w, screen_h, width, height

    def _clamp_chat_launcher_pos(self, x: int, y: int) -> tuple[int, int]:
        screen_w, screen_h, width, height = self._chat_launcher_metrics()
        max_x = max(0, screen_w - width)
        max_y = max(0, screen_h - height)
        return max(0, min(int(x), max_x)), max(0, min(int(y), max_y))

    def _handle_chat_launcher_click(self) -> None:
        if time.time() < float(self._chat_launcher_suppress_click_until):
            return
        self._open_chat_window()

    def _raise_chat_launcher(self, force_restack: bool = False) -> None:
        if self._chat_launcher is None or not self._chat_launcher.winfo_exists():
            return
        launcher = self._chat_launcher
        try:
            if force_restack:
                launcher.attributes("-topmost", False)
            launcher.attributes("-topmost", True)
        except Exception:
            pass
        try:
            if self._root is not None and self._root.winfo_exists():
                launcher.lift(self._root)
            else:
                launcher.lift()
        except Exception:
            pass

    def _position_chat_launcher(self) -> None:
        if self._chat_launcher is None or not self._chat_launcher.winfo_exists():
            return
        screen_w, screen_h, width, height = self._chat_launcher_metrics()
        top_margin = max(10, int(screen_h * 0.02))
        if self._chat_launcher_pos is None:
            x = max(0, (screen_w - width) // 2)
            y = top_margin
        else:
            x, y = self._clamp_chat_launcher_pos(*self._chat_launcher_pos)
            self._chat_launcher_pos = (x, y)
        self._chat_launcher.geometry(f"{width}x{height}+{x}+{y}")
        self._raise_chat_launcher()

    def _ensure_chat_launcher(self) -> None:
        assert self._root is not None
        if self._chat_launcher is not None and self._chat_launcher.winfo_exists():
            return
        launcher = ctk.CTkToplevel(self._root)
        launcher.overrideredirect(True)
        try:
            launcher.attributes("-topmost", True)
        except Exception:
            pass
        launcher.configure(fg_color="#2563EB")
        # UI POLISH ONLY
        btn = ctk.CTkButton(
            launcher,
            text="?? --:--:--",
            width=170,
            height=36,
            fg_color="#2563EB",
            hover_color="#1D4ED8",
            text_color="#F8FAFC",
            font=("Arial", 12, "bold"),
            corner_radius=12,
            border_width=0,
            command=self._handle_chat_launcher_click,
        )
        btn.pack(fill="both", expand=True, padx=0, pady=0)

        def _on_press(event) -> None:
            if launcher is None or not launcher.winfo_exists():
                return
            self._chat_launcher_drag_state = {
                "press_x": int(event.x_root),
                "press_y": int(event.y_root),
                "start_x": int(launcher.winfo_x()),
                "start_y": int(launcher.winfo_y()),
                "dragging": False,
            }
            self._raise_chat_launcher(force_restack=True)

        def _on_drag(event) -> None:
            if not self._chat_enabled:
                return
            state = self._chat_launcher_drag_state
            if not state or launcher is None or not launcher.winfo_exists():
                return
            dx = int(event.x_root) - int(state["press_x"])
            dy = int(event.y_root) - int(state["press_y"])
            if (not state["dragging"]) and max(abs(dx), abs(dy)) < 6:
                return
            state["dragging"] = True
            new_x = int(state["start_x"]) + dx
            new_y = int(state["start_y"]) + dy
            new_x, new_y = self._clamp_chat_launcher_pos(new_x, new_y)
            self._chat_launcher_pos = (new_x, new_y)
            launcher.geometry(f"+{new_x}+{new_y}")
            self._raise_chat_launcher()
            self._chat_launcher_suppress_click_until = time.time() + 0.25

        def _on_release(_event) -> None:
            state = self._chat_launcher_drag_state
            if state and state.get("dragging"):
                self._chat_launcher_suppress_click_until = time.time() + 0.25
            self._chat_launcher_drag_state = None
            self._raise_chat_launcher(force_restack=True)

        btn.bind("<ButtonPress-1>", _on_press, add="+")
        btn.bind("<B1-Motion>", _on_drag, add="+")
        btn.bind("<ButtonRelease-1>", _on_release, add="+")
        
        self._chat_launcher = launcher
        self._chat_launcher_button = btn
        self._position_chat_launcher()
        launcher.withdraw()

    def _set_chat_enabled_ui(self, enabled: bool) -> None:
        self._chat_enabled = bool(enabled)
        if not self._chat_enabled:
            if self._chat_window is not None and self._chat_window.winfo_exists():
                self._chat_window.destroy()
            self._chat_window = None
            self._chat_scroll = None
            if self._chat_launcher is not None and self._chat_launcher.winfo_exists():
                self._chat_launcher.withdraw()
            if self._root is not None and self._chat_timer_after_id is not None:
                try:
                    self._root.after_cancel(self._chat_timer_after_id)
                except Exception:
                    pass
                self._chat_timer_after_id = None
            return
        self._ensure_chat_launcher()
        if self._chat_launcher is not None and self._chat_launcher.winfo_exists():
            self._position_chat_launcher()
            self._update_chat_launcher_label()
            self._chat_launcher.deiconify()
            self._chat_launcher.lift()
            self._chat_launcher.attributes("-topmost", True)
        self._schedule_chat_launcher_label_update()

    def _append_chat_history(self, direction: str, text: str, ts: Optional[float] = None) -> None:
        payload = {
            "direction": direction,
            "text": str(text or "").strip(),
            "ts": float(ts) if ts is not None else time.time(),
        }
        if not payload["text"]:
            return
        self._chat_history.append(payload)
        if len(self._chat_history) > 500:
            del self._chat_history[:-500]

    def _refresh_chat_window(self) -> None:
        if self._chat_window is None or not self._chat_window.winfo_exists() or self._chat_scroll is None:
            return
        for child in self._chat_scroll.winfo_children():
            child.destroy()
        if not self._chat_history:
            ctk.CTkLabel(self._chat_scroll, text="No messages yet.", text_color="#94A3B8").pack(anchor="w", padx=8, pady=8)
            return
        for item in self._chat_history:
            direction = str(item.get("direction", ""))
            if direction == "teacher_to_student":
                who = "Teacher"
                anchor = "w"
            elif direction == "student_to_teacher":
                who = "You"
                anchor = "e"
            else:
                who = "System"
                anchor = "w"
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(item.get("ts", time.time()))))
            text = str(item.get("text", ""))
            row = ctk.CTkFrame(self._chat_scroll, fg_color="transparent")
            row.pack(fill="x", padx=8, pady=2)
            bubble = ctk.CTkFrame(row, fg_color="#1f6aa5", corner_radius=8, border_width=0)
            bubble.pack(anchor=anchor, padx=10, pady=6)
            ctk.CTkLabel(bubble, text=f"{who} - {ts}", text_color="#FFFFFF", font=("Arial", 11, "bold")).pack(anchor="w", padx=10, pady=(6, 0))
            ctk.CTkLabel(bubble, text=text, text_color="#FFFFFF", font=("Arial", 12), justify="left", wraplength=320).pack(anchor="w", padx=10, pady=(0, 6))

    def _open_chat_window(self) -> None:
        assert self._root is not None

        if not self._chat_enabled:
            return

        if self._chat_window is not None and self._chat_window.winfo_exists():
            self._chat_window.lift()
            self._chat_window.focus_force()
            return

        win = ctk.CTkToplevel(self._root)
        win.title("Chat")
        win.geometry("520x460")
        win.configure(fg_color="#0F172A")

        # important for lockscreen overlay
        win.attributes("-topmost", True)
        win.transient(self._root)

        win.lift()
        win.focus_force()

        self._chat_window = win
        self._position_chat_launcher()
        def _on_close() -> None:
            if win.winfo_exists():
                win.destroy()

            self._chat_window = None
            self._chat_scroll = None

            # restore chat launcher
            if self._chat_launcher is not None and self._chat_launcher.winfo_exists():
                self._chat_launcher.deiconify()
                self._chat_launcher.lift()
                self._chat_launcher.attributes("-topmost", True)

        win.protocol("WM_DELETE_WINDOW", _on_close)

        hdr = ctk.CTkFrame(win, fg_color="transparent")
        hdr.pack(fill="x", padx=10, pady=(10, 6))
        ctk.CTkLabel(hdr, text="Admin Conversation", text_color="#F8FAFC", font=("Arial", 14, "bold")).pack(side="left")

        scroll = ctk.CTkScrollableFrame(win, fg_color="#1E293B", border_width=1, border_color="#334155")
        scroll.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        self._chat_scroll = scroll

        compose = ctk.CTkFrame(win, fg_color="transparent")
        compose.pack(fill="x", padx=10, pady=(0, 10))
        entry = ctk.CTkEntry(compose, placeholder_text="Type message...", width=390)
        entry.pack(side="left", padx=(0, 6))

        def _send_now() -> None:
            text = entry.get().strip()
            if not text:
                return
            entry.delete(0, "end")
            self._append_chat_history("student_to_teacher", text)
            self._refresh_chat_window()

            def _worker() -> None:
                ok = bool(self._chat_send_handler(text)) if self._chat_send_handler else False
                if not ok:
                    self.append_chat_message_async("system", f"Send failed: {text}")

            threading.Thread(target=_worker, daemon=True).start()

        ctk.CTkButton(compose, text="Send", width=90, command=_send_now).pack(side="left")
        entry.bind("<Return>", lambda _e: _send_now())
        self._refresh_chat_window()
    def append_chat_message_async(self, direction: str, text: str, ts: Optional[float] = None) -> bool:
        self._ensure_worker()
        if not self._ui_ready_event.is_set():
            return False
        self._cmd_q.put(("chat_add", {"direction": direction, "text": text, "ts": ts}, threading.Event(), {"ok": False}))
        return True

    def _show_hidden(self) -> None:
        if self._auth_frame is not None and self._auth_frame.winfo_exists():
            self._auth_frame.place_forget()
        if self._lock_frame is not None and self._lock_frame.winfo_exists():
            self._lock_frame.place_forget()
        if self._connecting_frame is not None and self._connecting_frame.winfo_exists():
            self._connecting_frame.place_forget()
        self._hide_broadcast()

    def _apply_state(self, state: OverlayState, send_login=None, send_register=None) -> None:
        assert self._root is not None

        self._state = state

        if state == OverlayState.HIDDEN:
            self._show_hidden()
            self._root.attributes("-fullscreen", False)
            self._root.attributes("-topmost", False)
            try:
                self._root.state("normal")
            except Exception:
                pass
            self._root.after(0, self._root.withdraw)
            if self._chat_enabled:
                self._root.after(50, lambda: self._set_chat_enabled_ui(True))
            return

        elif state == OverlayState.LOCKED_TEMPORARY:
            self._show_lock()

        elif state == OverlayState.CONNECTING:
            self._show_connecting()

        elif state == OverlayState.AUTH_REQUIRED:
            if send_login is not None and send_register is not None:
                self._auth_handlers = (send_login, send_register)
            elif self._auth_handlers is not None:
                send_login, send_register = self._auth_handlers
            else:
                def _bootstrap_noop_login(_payload: dict) -> Optional[dict]:
                    return None

                def _bootstrap_noop_register(_payload: dict) -> Optional[dict]:
                    return None

                send_login, send_register = _bootstrap_noop_login, _bootstrap_noop_register
            self._show_auth(send_login, send_register)

        elif state == OverlayState.BROADCAST:
            self._show_broadcast()

        # HIDDEN FIX + FULLSCREEN FIX
        self._root.deiconify()
        self._apply_fullscreen_geometry()
        self._root.attributes("-fullscreen", True)
        self._root.attributes("-topmost", True)
        try:
            self._root.state("zoomed")
        except Exception:
            pass
        self._root.focus_force()
        if self._chat_enabled:
            self._set_chat_enabled_ui(True)


    def _ensure_toast_label(self) -> None:
        assert self._root is not None
        if self._toast_label is not None and self._toast_label.winfo_exists():
            return
        self._toast_label = ctk.CTkLabel(
            self._root,
            text="",
            fg_color="#0B5ED7",
            text_color="#FFFFFF",
            corner_radius=12,
            font=("Arial", 16, "bold"),
            padx=18,
            pady=10,
        )

    def _show_toast(self, text: str, duration_ms: int = 4500) -> None:
        assert self._root is not None
        message = str(text or "").strip()
        if not message:
            return
        self._ensure_toast_label()
        assert self._toast_label is not None
        self._toast_label.configure(text=message)
        self._toast_label.place(relx=0.5, y=24, anchor="n")
        self._toast_label.lift()

        if self._toast_after_id is not None:
            try:
                self._root.after_cancel(self._toast_after_id)
            except Exception:
                pass
            self._toast_after_id = None

        def _hide_toast() -> None:
            if self._toast_label is not None and self._toast_label.winfo_exists():
                self._toast_label.place_forget()
            self._toast_after_id = None

        self._toast_after_id = self._root.after(max(500, int(duration_ms)), _hide_toast)

    def _report_tk_callback_exception(self, exc, val, tb) -> None:
        detail = "".join(traceback.format_exception(exc, val, tb))
        print(detail)
        try:
            self._show_toast("Overlay recovered after a UI issue.", duration_ms=5000)
        except Exception:
            pass

    def _apply_fullscreen_geometry(self) -> None:
        assert self._root is not None
        screen_w = max(1, int(self._root.winfo_screenwidth()))
        screen_h = max(1, int(self._root.winfo_screenheight()))
        self._root.geometry(f"{screen_w}x{screen_h}+0+0")
        try:
            self._root.attributes("-fullscreen", True)
        except Exception:
            pass

    def _ui_loop(self) -> None:
        try:
            self._root = ctk.CTk()
            self._root.report_callback_exception = self._report_tk_callback_exception
            self._ui_ready_event.set()

            self._root.title("Student Overlay")
            self._root.protocol("WM_DELETE_WINDOW", lambda: None)
            self._root.configure(fg_color="#0e1116")
            self._apply_fullscreen_geometry()
            self._root.bind("<Alt-F4>", lambda _e: "break")
            if DEV_MODE:
                self._root.attributes("-fullscreen", False)
                self._root.attributes("-topmost", False)
                def _dev_force_exit(_event=None):
                    if self._root is None:
                        return
                    try:
                        self._root.attributes("-fullscreen", False)
                    except Exception:
                        pass
                    try:
                        self._root.attributes("-topmost", False)
                    except Exception:
                        pass
                    try:
                        self._root.state("normal")
                    except Exception:
                        pass
                    try:
                        self._root.destroy()
                    except Exception:
                        pass
                    os._exit(0)

                self._root.bind_all("<Control-Shift-D>", _dev_force_exit)
               

            def pump_commands() -> None:
                try:
                    while True:
                        try:
                            cmd, payload, done, result = self._cmd_q.get_nowait()
                        except queue.Empty:
                            break
                        try:
                            if cmd == "set_state":
                                self._apply_state(payload["state"])
                                result["ok"] = True
                            elif cmd == "auth":
                                while not self._auth_result_q.empty():
                                    try:
                                        self._auth_result_q.get_nowait()
                                    except queue.Empty:
                                        break
                                self._apply_state(
                                    OverlayState.AUTH_REQUIRED,
                                    payload["send_login"],
                                    payload["send_register"],
                                )
                                result["ok"] = True
                            elif cmd == "toast":
                                self._show_toast(payload.get("text", ""), int(payload.get("duration_ms", 4500)))
                                result["ok"] = True
                            elif cmd == "chat_add":
                                self._append_chat_history(str(payload.get("direction", "teacher_to_student")), payload.get("text", ""), payload.get("ts"))
                                self._refresh_chat_window()
                                result["ok"] = True
                            elif cmd == "chat_visibility":
                                self._set_chat_enabled_ui(bool(payload.get("enabled", False)))
                                result["ok"] = True
                            elif cmd == "broadcast_frame":
                                generation = payload.get("generation")
                                with self._broadcast_generation_lock:
                                    current_generation = self._broadcast_generation
                                if generation != current_generation:
                                    # A frame from a prior TCP downlink must not
                                    # repaint the overlay after Stop/Start.
                                    result["ok"] = True
                                    continue
                                self._emit_broadcast_diag("ui_frame_dequeued", queue_size=self._cmd_q.qsize())
                                self._render_broadcast_frame(payload.get("image"))
                                result["ok"] = True
                            elif cmd == "broadcast_clear":
                                self._clear_broadcast_ui()
                                result["ok"] = True
                            else:
                                result["ok"] = False
                        except Exception:
                            result["ok"] = False
                        finally:
                            done.set()
                except Exception:
                    exc_type, exc_val, exc_tb = sys.exc_info()
                    if exc_type is not None:
                        self._report_tk_callback_exception(exc_type, exc_val, exc_tb)
                finally:
                    if self._root is not None:
                        try:
                            self._root.after(50, pump_commands)
                        except Exception:
                            pass

            self._root.after(50, pump_commands)
            self._root.mainloop()
        finally:
            self._ui_ready_event.clear()
            self._root = None
            self._lock_frame = None
            self._auth_frame = None
            self._connecting_frame = None
            self._broadcast_frame = None
            self._broadcast_image_label = None
            self._broadcast_status_label = None
            self._broadcast_photo = None
            self._toast_label = None
            self._chat_window = None
            self._chat_scroll = None
            self._chat_launcher = None
            self._chat_launcher_button = None
            self._toast_after_id = None
            self._chat_timer_after_id = None
            with self._worker_lock:
                self._worker_started = False
                self._worker_thread = None

    def set_state(self, state: OverlayState, timeout_s: float = 2.0) -> bool:
        self._ensure_worker()

        # NEW: first bootstrap must wait deterministically for UI root
        if not self._ui_ready_event.is_set():
            if not self._ui_ready_event.wait(timeout=timeout_s):
                return False

        done = threading.Event()
        result = {"ok": False}
        self._cmd_q.put(("set_state", {"state": state}, done, result))

        # After UI ready, state application must complete deterministically
        if not done.wait(timeout=timeout_s):
            return False

        return bool(result.get("ok"))

    def set_chat_enabled(self, enabled: bool, timeout_s: float = 1.0) -> bool:
        self._ensure_worker()
        if not self._ui_ready_event.is_set():
            if not self._ui_ready_event.wait(timeout=timeout_s):
                return False
        done = threading.Event()
        result = {"ok": False}
        self._cmd_q.put(("chat_visibility", {"enabled": bool(enabled)}, done, result))
        if not done.wait(timeout=timeout_s):
            return False
        return bool(result.get("ok"))

    def notify_message(self, text: str, duration_ms: int = 4500, timeout_s: float = 1.0) -> bool:
        self._ensure_worker()
        if not self._ui_ready_event.is_set():
            if not self._ui_ready_event.wait(timeout=timeout_s):
                return False
        done = threading.Event()
        result = {"ok": False}
        self._cmd_q.put(("toast", {"text": text, "duration_ms": int(duration_ms)}, done, result))
        if not done.wait(timeout=timeout_s):
            return False
        return bool(result.get("ok"))

    def notify_message_async(self, text: str, duration_ms: int = 4500) -> bool:
        self._ensure_worker()
        if not self._ui_ready_event.is_set():
            return False
        self._cmd_q.put(("toast", {"text": text, "duration_ms": int(duration_ms)}, threading.Event(), {"ok": False}))
        return True

    def authenticate(self, send_login, send_register, timeout_s: float = 300.0) -> Optional[dict]:
        self._ensure_worker()

        done = threading.Event()
        result = {"ok": False}
        self._cmd_q.put(("auth", {"send_login": send_login, "send_register": send_register}, done, result))

        if not done.wait(timeout=5.0):
            return None
        if not result.get("ok"):
            return None
        try:
            return self._auth_result_q.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def set_broadcast_generation(self, generation: int) -> None:
        """Invalidate queued frames that do not belong to *generation*."""
        with self._broadcast_generation_lock:
            self._broadcast_generation = generation

    def show_broadcast_frame_async(self, image: Image.Image, generation: int) -> bool:
        self._ensure_worker()
        if not self._ui_ready_event.is_set():
            return False
        self._cmd_q.put(("broadcast_frame", {"image": image, "generation": generation}, threading.Event(), {"ok": False}))
        return True

    def clear_broadcast_frame_async(self) -> bool:
        self._ensure_worker()
        if not self._ui_ready_event.is_set():
            return False
        self._cmd_q.put(("broadcast_clear", {}, threading.Event(), {"ok": False}))
        return True


class TimerManager:
    def __init__(self, on_lock, on_expire, on_warning=None, warning_enabled: bool = False) -> None:
        self.on_lock = on_lock
        self.on_expire = on_expire
        self.on_warning = on_warning
        self.warning_enabled = warning_enabled
        self.offset = 0.0
        self.lock = threading.Lock()
        self.timers: dict[str, dict] = {}
        threading.Thread(target=self._tick_loop, daemon=True).start()

    def sync_time(self, server_ts: float) -> None:
        self.offset = server_ts - time.time()

    def set_timer(self, payload: dict) -> None:
        with self.lock:
            timer = dict(payload)
            timer.setdefault("warning_sent", False)
            timer.setdefault("paused", False)
            timer.setdefault("paused_at_ts", None)
            timer.setdefault("paused_accum_ms", 0)
            self.timers[payload["timer_id"]] = timer

    def extend_timer(self, timer_id: str, extra_ms: int) -> None:
        with self.lock:
            timer = self.timers.get(timer_id)
            if timer:
                timer["duration_ms"] += extra_ms

    def cancel_timer(self, timer_id: str) -> None:
        with self.lock:
            if timer_id in self.timers:
                self.timers[timer_id]["cancelled"] = True

    def pause_timer(self, timer_id: str, pause_ts: Optional[float] = None) -> None:
        with self.lock:
            timer = self.timers.get(timer_id)
            if not timer or timer.get("cancelled") or timer.get("expired"):
                return
            if timer.get("paused"):
                return
            timer["paused"] = True
            timer["paused_at_ts"] = float(pause_ts) if pause_ts is not None else (time.time() + self.offset)
            timer.setdefault("paused_accum_ms", 0)

    def resume_timer(self, timer_id: str, resume_ts: Optional[float] = None) -> None:
        with self.lock:
            timer = self.timers.get(timer_id)
            if not timer or timer.get("cancelled") or timer.get("expired"):
                return
            if not timer.get("paused"):
                return
            now_ts = float(resume_ts) if resume_ts is not None else (time.time() + self.offset)
            paused_at = timer.get("paused_at_ts")
            if paused_at is not None:
                timer["paused_accum_ms"] = int(timer.get("paused_accum_ms", 0)) + max(0, int((now_ts - float(paused_at)) * 1000))
            timer["paused"] = False
            timer["paused_at_ts"] = None

    def _tick_loop(self) -> None:
        while True:
            now = time.time() + self.offset

            expired_ids: list[str] = []
            remove_ids: list[str] = []

            with self.lock:
                for timer_id, timer in list(self.timers.items()):
                    if timer.get("cancelled"):
                        remove_ids.append(timer_id)
                        continue

                    elapsed = int((now - timer["start_ts"]) * 1000)
                    paused_ms = int(timer.get("paused_accum_ms", 0))
                    if timer.get("paused") and timer.get("paused_at_ts") is not None:
                        paused_ms += max(0, int((now - float(timer.get("paused_at_ts"))) * 1000))
                    effective_elapsed = max(0, elapsed - paused_ms)
                    remaining = timer["duration_ms"] - effective_elapsed

                    warning_ms = int(timer.get("warning_ms", 0))
                    if (
                        self.warning_enabled
                        and self.on_warning is not None
                        and warning_ms > 0
                        and remaining > 0
                        and remaining <= warning_ms
                        and not timer.get("warning_sent")
                    ):
                        timer["warning_sent"] = True
                        try:
                            self.on_warning(timer_id, remaining)
                        except Exception:
                            pass

                    if remaining <= 0 and not timer.get("expired"):
                        timer["expired"] = True
                        expired_ids.append(timer_id)
                        remove_ids.append(timer_id)

            for timer_id in expired_ids:
                self.on_lock()
                self.on_expire(timer_id)

            with self.lock:
                for timer_id in remove_ids:
                    self.timers.pop(timer_id, None)

            time.sleep(RUNTIME.timer_tick_s)


class StudentDeployClient:
    def __init__(self, teacher_ip: str, settings_store: Optional[StudentSettingsStore] = None) -> None:
        self.settings_store = settings_store or StudentSettingsStore(default_teacher_host=teacher_ip)
        self.teacher_ip = normalize_teacher_host(teacher_ip)
        self.hostname = socket.gethostname()
        self.mac = self._get_mac()
        self.pc_id: Optional[str] = None

        self.overlay = OverlayController()
        self.overlay.set_broadcast_diag_callback(self._on_overlay_broadcast_diag)
        self.overlay.set_chat_sender(self.send_session_message)
        self.overlay.set_teacher_host_settings(self.get_teacher_ip, self.save_teacher_ip)
        self.overlay.set_chat_timer_provider(self._chat_remaining_s)
        self.timer_manager = TimerManager(
            self._on_timer_lock,
            self._on_timer_expired,
            on_warning=self._on_timer_warning,
            warning_enabled=bool(getattr(RUNTIME, "timer_tick_s", 0) >= 0 and False),
        )

        self.control_sock: Optional[socket.socket] = None
        self.video_sock: Optional[socket.socket] = None
        self.broadcast_sock: Optional[socket.socket] = None
        self.broadcast_audio_sock: Optional[socket.socket] = None
        self.control_file = None
        self.send_lock = threading.Lock()
        self.conn_lock = threading.Lock()
        self.cmd_lock = threading.Lock()
        self.seen_cmd_ids: deque[str] = deque(maxlen=512)
        self.connected = False
        self.connecting = False
        self.boot_ts = psutil.boot_time()
        self.current_user: Optional[dict] = None
        self.stream_width = RUNTIME.frame_width
        self.stream_height = RUNTIME.frame_height
        self.jpeg_quality = 80
        self.stream_fps = max(1, RUNTIME.max_fps)
        self.dynamic_reconnect_interval_s = RUNTIME.reconnect_interval_s
        self.auth_response_queue: "queue.Queue[dict]" = queue.Queue()
        self.temporary_lock_active = False
        self.signout_lock_active = False
        self.enable_session_messaging = False
        self.enable_extension_requests = False
        self.enable_timer_near_limit_notify = False
        self.enable_timer_pause_on_temp_lock = False
        self.pending_extension_offers: dict[str, dict] = {}
        self.session_timer_id = "session"
        self.session_extension_ms_by_timer: dict[str, int] = {}
        self.state_lock = threading.Lock()
        self.udp_send_sock: Optional[socket.socket] = None
        self.udp_send_lock = threading.Lock()
        self._showing_connection_overlay = False
        self.broadcast_active = False
        # Monotonically identifies both active sessions and invalidated sessions.
        # It is separate from the diagnostic counter so receiver ownership checks
        # are synchronized with broadcast_active.
        self.broadcast_generation = 0
        self._broadcast_audio_missing_warned = False
        self._broadcast_audio_device_warned = False
        self.logger = _student_logger()
        self.broadcast_diag_counters = {
            "frames_received": 0,
            "frames_decoded": 0,
            "frames_enqueued": 0,
            "frames_dequeued": 0,
            "frames_rendered": 0,
        }
        self.broadcast_diag_first_frame_mono: Optional[float] = None
        self.broadcast_diag_last_render_mono: Optional[float] = None
        self.broadcast_diag_session = 0

    def _broadcast_diag(self, event: str, **data: object) -> None:
        with self.state_lock:
            active = bool(self.broadcast_active)
            connected = bool(self.connected)
            connecting = bool(self.connecting)
        with self.conn_lock:
            broadcast_sock = self.broadcast_sock
            control_sock = self.control_sock
        payload = {
            "ts": round(time.time(), 3),
            "mono_ts": round(time.monotonic(), 6),
            "event": "broadcast_diag",
            "diag_event": event,
            "side": "STUDENT",
            "thread_name": threading.current_thread().name,
            "thread_ident": threading.get_ident(),
            "pc_id": self.pc_id,
            "broadcast_diag_session": self.broadcast_diag_session,
            "broadcast_active": active,
            "connected": connected,
            "connecting": connecting,
            "broadcast_sock_id": id(broadcast_sock) if broadcast_sock else None,
            "control_sock_id": id(control_sock) if control_sock else None,
            "counters": dict(self.broadcast_diag_counters),
            **data,
        }
        self.logger.info(json.dumps(payload, sort_keys=True))

    def _on_overlay_broadcast_diag(self, event: str, data: dict) -> None:
        should_log = False
        if event == "ui_frame_dequeued":
            self.broadcast_diag_counters["frames_dequeued"] += 1
            should_log = self.broadcast_diag_counters["frames_dequeued"] == 1 or self.broadcast_diag_counters["frames_dequeued"] % 100 == 0
        elif event == "ui_rendered":
            self.broadcast_diag_counters["frames_rendered"] += 1
            self.broadcast_diag_last_render_mono = time.monotonic()
            should_log = self.broadcast_diag_counters["frames_rendered"] == 1 or self.broadcast_diag_counters["frames_rendered"] % 100 == 0
        if should_log:
            self._broadcast_diag(event, queue_size=data.get("queue_size"), last_render_mono=self.broadcast_diag_last_render_mono)

    def get_teacher_ip(self) -> str:
        return self.teacher_ip

    def save_teacher_ip(self, new_host: str) -> tuple[bool, str]:
        try:
            normalized = normalize_teacher_host(new_host)
            changed = self.update_teacher_ip(normalized)
            self.settings_store.save(StudentSettings(teacher_host=normalized))
            if changed:
                return True, f"Saved. Reconnecting to {normalized}..."
            return True, f"Saved. Already using {normalized}."
        except ValueError as exc:
            return False, str(exc)
        except Exception as exc:
            return False, f"Could not save server IP: {exc}"

    def update_teacher_ip(self, new_host: str) -> bool:
        normalized = normalize_teacher_host(new_host)
        if normalized == self.teacher_ip:
            return False
        self.teacher_ip = normalized
        self._update_state(connected=False, connecting=False)
        self._cleanup_sockets()
        return True


    def _state_snapshot(self) -> dict:
        with self.state_lock:
            return {
                "connected": bool(self.connected),
                "connecting": bool(self.connecting),
                "current_user": self.current_user,
                "signout_lock_active": bool(self.signout_lock_active),
                "temporary_lock_active": bool(self.temporary_lock_active),
                "broadcast_active": bool(self.broadcast_active),
                "broadcast_generation": int(self.broadcast_generation),
            }

    def _update_state(self, **changes) -> None:
        with self.state_lock:
            for key, value in changes.items():
                setattr(self, key, value)

    def _should_show_connection_overlay(self, snapshot: Optional[dict] = None) -> bool:
        current = snapshot if snapshot is not None else self._state_snapshot()
        return (not current["current_user"]) and (not current["temporary_lock_active"])

    def _show_connection_wait_ui(self) -> None:
        snapshot = self._state_snapshot()
        if not self._should_show_connection_overlay(snapshot):
            self._showing_connection_overlay = False
            return
        if self._showing_connection_overlay:
            return
        if self.overlay.set_state(OverlayState.CONNECTING, timeout_s=0.5):
            self._showing_connection_overlay = True

    def _chat_remaining_s(self) -> Optional[int]:
        snapshot = self._state_snapshot()
        if snapshot["signout_lock_active"] or not snapshot["current_user"]:
            return None
        return self._timer_remaining_s(self.session_timer_id)

    def _set_session_timer_from_max_session(self, max_session_s: object, server_ts: float) -> None:
        if not isinstance(max_session_s, (int, float)):
            return

        duration_ms = max(0, int(max_session_s) * 1000)

        payload = {
            "timer_id": self.session_timer_id,
            "start_ts": float(server_ts),   # ? server time (same as admin)
            "duration_ms": duration_ms,
            "warning_ms": 0,
            "paused": False,
            "paused_at_ts": None,
            "paused_accum_ms": 0,
            "action": "SESSION",
        }

        self.session_extension_ms_by_timer.clear()
        self.timer_manager.set_timer(payload)

    def _clear_session_timer(self) -> None:
        self.timer_manager.cancel_timer(self.session_timer_id)
        with self.timer_manager.lock:
            self.timer_manager.timers.pop(self.session_timer_id, None)
        self.session_extension_ms_by_timer.clear()

    def _adjust_session_timer_duration(self, delta_ms: int) -> bool:
        with self.timer_manager.lock:
            timer = self.timer_manager.timers.get(self.session_timer_id)
            if not timer or timer.get("cancelled") or timer.get("expired"):
                return False
            timer["duration_ms"] = max(0, int(timer.get("duration_ms", 0)) + int(delta_ms))
            return True

    def _timer_remaining_s(self, timer_id: str) -> Optional[int]:
        with self.timer_manager.lock:
            timer = self.timer_manager.timers.get(timer_id)
            if not timer or timer.get("cancelled") or timer.get("expired"):
                return None
            now = time.time() + self.timer_manager.offset
            elapsed = int((now - float(timer.get("start_ts", now))) * 1000)
            paused_ms = int(timer.get("paused_accum_ms", 0))
            paused_at_ts = timer.get("paused_at_ts")
            if timer.get("paused") and paused_at_ts is not None:
                paused_ms += max(0, int((now - float(paused_at_ts)) * 1000))
            effective_elapsed = max(0, elapsed - paused_ms)
            remaining_ms = max(0, int(timer.get("duration_ms", 0)) - effective_elapsed)
            return remaining_ms // 1000

    def _on_timer_lock(self) -> None:
        with self.state_lock:
            if self.signout_lock_active:
                return
            self.temporary_lock_active = True
        self.overlay.set_state(OverlayState.LOCKED_TEMPORARY)

    def _restore_overlay_state(self) -> bool:
        snapshot = self._state_snapshot()
        self._showing_connection_overlay = False
        chat_enabled = bool(self.enable_session_messaging and snapshot["current_user"] and (not snapshot["signout_lock_active"]))
        self.overlay.set_chat_enabled(chat_enabled)
        if snapshot["broadcast_active"]:
            return self.overlay.set_state(OverlayState.BROADCAST)
        if snapshot["signout_lock_active"] or not snapshot["current_user"]:
            return self.overlay.set_state(OverlayState.AUTH_REQUIRED)
        if snapshot["temporary_lock_active"]:
            return self.overlay.set_state(OverlayState.LOCKED_TEMPORARY)
        return self.overlay.set_state(OverlayState.HIDDEN)

    def _get_mac(self) -> str:
        mac_int = uuid.getnode()
        return ":".join(f"{(mac_int >> shift) & 0xff:02x}" for shift in range(40, -1, -8))

    def _safe_send_json(self, sock: Optional[socket.socket], payload: dict) -> bool:
        if sock is None:
            return False
        with self.send_lock:
            try:
                send_json(sock, payload)
                return True
            except (OSError, AttributeError):
                return False

    def _is_duplicate_command(self, cmd_id: str) -> bool:
        if not cmd_id:
            return False
        with self.cmd_lock:
            if cmd_id in self.seen_cmd_ids:
                return True
            self.seen_cmd_ids.append(cmd_id)
        return False

    def _send_udp_json(self, payload: dict) -> bool:
        data = json.dumps(payload).encode("utf-8")
        with self.udp_send_lock:
            if self.udp_send_sock is None:
                try:
                    self.udp_send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                except OSError:
                    self.udp_send_sock = None
                    return False
            try:
                self.udp_send_sock.sendto(data, (self.teacher_ip, NETWORK.command_fallback_port))
                return True
            except OSError:
                try:
                    if self.udp_send_sock is not None:
                        self.udp_send_sock.close()
                except OSError:
                    pass
                self.udp_send_sock = None
                return False

    def _send_ack(self, ack: dict) -> None:
        is_broadcast_ack = ack.get("command") in {"BROADCAST_START", "BROADCAST_STOP"}
        if is_broadcast_ack:
            self._broadcast_diag("command_ack_attempt", command=ack.get("command"), cmd_id=ack.get("cmd_id"), result=ack.get("result"), reason=ack.get("reason", ""), note="ack_does_not_prove_downlink_registration")
        with self.conn_lock:
            control_sock = self.control_sock
        if control_sock and self.pc_id:
            if self._safe_send_json(control_sock, ack):
                if is_broadcast_ack:
                    self._broadcast_diag("command_ack_sent", command=ack.get("command"), cmd_id=ack.get("cmd_id"), transport="tcp")
                return
        self._send_udp_json(ack)
        if is_broadcast_ack:
            self._broadcast_diag("command_ack_sent", command=ack.get("command"), cmd_id=ack.get("cmd_id"), transport="udp_fallback")

    def _make_ack(self, command: object, cmd_id: str, applied: bool, reason: str = "") -> dict:
        return {
            "type": "ack",
            "pc_id": self.pc_id,
            "command": command,
            "cmd_id": cmd_id,
            "result": "applied" if applied else "failed",
            "reason": reason,
        }

    def _start_broadcast_session(self) -> bool:
        with self.state_lock:
            self.broadcast_generation += 1
            generation = self.broadcast_generation
            self.broadcast_active = True
        self.broadcast_diag_session = generation
        self.broadcast_diag_first_frame_mono = None
        self._broadcast_diag("student_start_session_enter")
        self.overlay.set_broadcast_generation(generation)
        self.overlay.clear_broadcast_frame_async()
        applied = self.overlay.set_state(OverlayState.BROADCAST)
        if not applied:
            with self.state_lock:
                if self.broadcast_generation == generation:
                    self.broadcast_active = False
        self._broadcast_diag("student_start_session_complete", overlay_applied=applied)
        return applied

    def _stop_broadcast_session(self) -> bool:
        self._broadcast_diag("student_stop_session_enter")
        # Invalidate before closing the socket: recv_frame() may return after a
        # new Start has already installed a replacement socket.
        with self.state_lock:
            self.broadcast_generation += 1
            generation = self.broadcast_generation
            self.broadcast_active = False
        self.broadcast_diag_session = generation
        self.overlay.set_broadcast_generation(generation)
        self._close_broadcast_socket()
        self._close_broadcast_audio_socket()
        self.overlay.clear_broadcast_frame_async()
        applied = self._restore_overlay_state()
        self._broadcast_diag("student_stop_session_complete", overlay_applied=applied)
        return applied

    def _run_power_command(self, action: str) -> None:
        system_name = platform.system().strip().lower()
        if system_name != "windows":
            return
        if action == "SHUTDOWN":
            os.system("shutdown /s /t 0")
        elif action == "RESTART":
            os.system("shutdown /r /t 0")

    def _schedule_power_command(self, action: str):
        def _runner() -> None:
            threading.Thread(target=self._run_power_command, args=(action,), daemon=True).start()
        return _runner

    def _execute_command(self, msg: dict, *, from_udp: bool = False):
        command = msg.get("command")
        cmd_id = str(msg.get("cmd_id", "")).strip()
        if command in {"BROADCAST_START", "BROADCAST_STOP"}:
            self._broadcast_diag("command_received", command=command, cmd_id=cmd_id, transport="udp" if from_udp else "tcp")
        if self._is_duplicate_command(cmd_id):
            return self._make_ack(command, cmd_id, True, "duplicate_ignored"), None
        applied = True
        reason = ""
        post_action = None
        if command == "LOCK_NOW":
            lock_mode = str(msg.get("lock_mode", "temporary")).strip().lower()
            if lock_mode == "signout":
                pause_ts = msg.get("pause_ts")
                if isinstance(pause_ts, (int, float)):
                    self.timer_manager.pause_timer(self.session_timer_id, float(pause_ts))
                else:
                    self.timer_manager.pause_timer(self.session_timer_id)
                self._update_state(current_user=None, signout_lock_active=True, temporary_lock_active=False)
                applied = self._restore_overlay_state()
            else:
                self._update_state(signout_lock_active=False, temporary_lock_active=True)
                applied = self._restore_overlay_state()
            if not applied:
                reason = "overlay_lock_failed"
        elif command == "UNLOCK_NOW":
            self._update_state(signout_lock_active=False, temporary_lock_active=False)
            applied = self._restore_overlay_state()
            if not applied:
                reason = "overlay_unlock_failed"
        elif command == "SET_TIMER":
            try:
                self.timer_manager.set_timer(msg)
            except Exception:
                applied = False
                reason = "set_timer_failed"
        elif command == "EXTEND_TIMER":
            try:
                timer_id = str(msg.get("timer_id", ""))
                extra_ms = int(msg.get("extra_ms", 0))
                self.timer_manager.extend_timer(timer_id, extra_ms)
                if timer_id != self.session_timer_id and extra_ms != 0:
                    if self._adjust_session_timer_duration(extra_ms):
                        self.session_extension_ms_by_timer[timer_id] = int(self.session_extension_ms_by_timer.get(timer_id, 0)) + extra_ms
            except Exception:
                applied = False
                reason = "extend_timer_failed"
        elif command == "CANCEL_TIMER":
            try:
                timer_id = str(msg.get("timer_id", ""))
                self.timer_manager.cancel_timer(timer_id)
                if timer_id and timer_id != self.session_timer_id:
                    rollback_ms = int(self.session_extension_ms_by_timer.pop(timer_id, 0))
                    if rollback_ms != 0:
                        self._adjust_session_timer_duration(-rollback_ms)
            except Exception:
                applied = False
                reason = "cancel_timer_failed"
        elif command == "SET_STREAM_PROFILE":
            try:
                width = int(msg.get("width", self.stream_width))
                height = int(msg.get("height", self.stream_height))
                jpeg_quality = int(msg.get("jpeg_quality", self.jpeg_quality))
                fps = int(msg.get("max_fps", self.stream_fps))
                self.stream_width = max(320, min(1920, width))
                self.stream_height = max(180, min(1080, height))
                self.jpeg_quality = max(40, min(95, jpeg_quality))
                self.stream_fps = max(1, min(30, fps))
            except Exception:
                applied = False
                reason = "set_stream_profile_failed"
        elif command == "SET_RUNTIME_TUNING":
            try:
                reconnect_s = int(msg.get("reconnect_interval_s", self.dynamic_reconnect_interval_s))
                self.dynamic_reconnect_interval_s = max(1, min(30, reconnect_s))
            except Exception:
                applied = False
                reason = "set_runtime_tuning_failed"
        elif command == "PAUSE_TIMER":
            try:
                if self.enable_timer_pause_on_temp_lock:
                    timer_id = str(msg.get("timer_id", ""))
                    pause_ts = msg.get("pause_ts")
                    self.timer_manager.pause_timer(timer_id, pause_ts)
                    if timer_id != self.session_timer_id:
                        self.timer_manager.pause_timer(self.session_timer_id, pause_ts)
            except Exception:
                applied = False
                reason = "pause_timer_failed"
        elif command == "RESUME_TIMER":
            try:
                if self.enable_timer_pause_on_temp_lock:
                    timer_id = str(msg.get("timer_id", ""))
                    resume_ts = msg.get("resume_ts")
                    self.timer_manager.resume_timer(timer_id, resume_ts)
                    if timer_id != self.session_timer_id:
                        self.timer_manager.resume_timer(self.session_timer_id, resume_ts)
            except Exception:
                applied = False
                reason = "resume_timer_failed"
        elif command == "SESSION_MESSAGE":
            try:
                if self.enable_session_messaging:
                    text = str(msg.get("text", "")).strip()
                    if text:
                        self.overlay.append_chat_message_async("teacher_to_student", text)
                        queued = self.overlay.notify_message_async(text)
                        if not queued:
                            print(f"[Admin Message] {text}")
            except Exception:
                applied = False
                reason = "session_message_failed"
        elif command == "EXTENSION_OFFER":
            try:
                if not self.enable_extension_requests:
                    applied = False
                    reason = "extension_requests_disabled"
                else:
                    request_id = str(msg.get("request_id", "")).strip()
                    approved = bool(msg.get("approved", False))
                    extra_ms = int(msg.get("extra_ms", 0))
                    if request_id and approved and extra_ms > 0:
                        self.pending_extension_offers[request_id] = {"extra_ms": extra_ms, "approved": approved}
                        self._safe_send_json(self.control_sock, {
                            "type": "extension_offer_response",
                            "request_id": request_id,
                            "decision": "accepted",
                        })
                    elif request_id and (not approved):
                        self._safe_send_json(self.control_sock, {
                            "type": "extension_offer_response",
                            "request_id": request_id,
                            "decision": "declined",
                        })
            except Exception:
                applied = False
                reason = "extension_offer_failed"
        elif command == "BROADCAST_START":
            try:
                applied = self._start_broadcast_session()
                if not applied:
                    reason = "broadcast_start_failed"
            except Exception:
                applied = False
                reason = "broadcast_start_failed"
        elif command == "BROADCAST_STOP":
            try:
                applied = self._stop_broadcast_session()
                if not applied:
                    reason = "broadcast_stop_failed"
            except Exception:
                applied = False
                reason = "broadcast_stop_failed"
        elif command == "SHUTDOWN":
            if from_udp:
                applied = False
                reason = "tcp_required"
            elif platform.system().strip().lower() != "windows":
                applied = False
                reason = "unsupported_platform"
            else:
                post_action = self._schedule_power_command("SHUTDOWN")
        elif command == "RESTART":
            if from_udp:
                applied = False
                reason = "tcp_required"
            elif platform.system().strip().lower() != "windows":
                applied = False
                reason = "unsupported_platform"
            else:
                post_action = self._schedule_power_command("RESTART")
        else:
            applied = False
            reason = "unsupported_command"
        ack = self._make_ack(command, cmd_id, applied, reason)
        if command in {"BROADCAST_START", "BROADCAST_STOP"}:
            self._broadcast_diag("command_executed", command=command, cmd_id=cmd_id, applied=applied, reason=reason, note="start_ack_does_not_prove_downlink_registration")
        return ack, post_action

    def _udp_fallback_loop(self) -> None:
        while True:
            sock: Optional[socket.socket] = None
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.bind(("0.0.0.0", NETWORK.command_fallback_port))
                while True:
                    data, _ = sock.recvfrom(4096)
                    try:
                        msg = json.loads(data.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue
                    if msg.get("type") != "command":
                        continue
                    ack, post_action = self._execute_command(msg, from_udp=True)
                    self._send_ack(ack)
                    if post_action is not None:
                        post_action()
            except OSError:
                time.sleep(1)
            except Exception:
                time.sleep(1)
            finally:
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass

    def _send_auth_message(self, payload: dict, expected_action: str, timeout_s: float = 15.0) -> Optional[dict]:
        with self.conn_lock:
            sock = self.control_sock
        if not sock:
            return None
        while not self.auth_response_queue.empty():
            try:
                self.auth_response_queue.get_nowait()
            except queue.Empty:
                break
        if not self._safe_send_json(sock, payload):
            self._update_state(connected=False)
            self._cleanup_sockets()
            return None
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            remaining = max(0.1, deadline - time.time())
            try:
                resp = self.auth_response_queue.get(timeout=remaining)
            except queue.Empty:
                continue
            if resp.get("type") != "auth_ack":
                continue
            if resp.get("action") != expected_action:
                continue
            if not isinstance(resp.get("ok"), bool):
                continue
            return resp
        return None

    def _authenticate_student(self) -> bool:
        snapshot = self._state_snapshot()
        current_user = snapshot["current_user"]
        signout_lock_active = snapshot["signout_lock_active"]
        if current_user and (not signout_lock_active):
            resp = self._send_auth_message({"type": "student_session_resume", "user": current_user}, "resume")
            if resp and resp.get("ok"):
                return self._restore_overlay_state()
            self._update_state(current_user=None)
            self._clear_session_timer()

        def send_login(payload: dict) -> Optional[dict]:
            return self._send_auth_message({"type": "student_login", **payload}, "login")

        def send_register(payload: dict) -> Optional[dict]:
            return self._send_auth_message({"type": "student_register", **payload}, "register")

        while True:
            user = self.overlay.authenticate(send_login, send_register)
            if user is None:
                return False
            self._update_state(current_user=user, signout_lock_active=False)
            return self._restore_overlay_state()

    def _connect_control(self) -> bool:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.teacher_ip, NETWORK.control_port))
            file_obj = sock.makefile("rb")
            with self.conn_lock:
                self.control_sock = sock
                self.control_file = file_obj
            send_json(sock, {"type": "register", "mac": self.mac, "hostname": self.hostname})
            ack = recv_json_line(file_obj)
            if not ack or ack.get("type") != "register_ack":
                self._cleanup_sockets()
                return False
            self.pc_id = str(ack.get("pc_id"))
            self.enable_session_messaging = bool(ack.get("enable_session_messaging", False))
            self.enable_extension_requests = bool(ack.get("enable_extension_requests", False))
            self.enable_timer_near_limit_notify = bool(ack.get("enable_timer_near_limit_notify", False))
            self.enable_timer_pause_on_temp_lock = bool(ack.get("enable_timer_pause_on_temp_lock", False))
            self.timer_manager.warning_enabled = self.enable_timer_near_limit_notify
            server_ts = ack.get("server_ts")
            if isinstance(server_ts, (int, float)):
                self.timer_manager.sync_time(float(server_ts))
            for timer in ack.get("timers", []):
                self.timer_manager.set_timer(timer)
            if not self._authenticate_student():
                self._cleanup_sockets()
                return False
            return True
        except (OSError, ValueError):
            self._cleanup_sockets()
            return False

    def _connect_video(self) -> bool:
        if not self.pc_id:
            return False
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.teacher_ip, NETWORK.video_port))
            send_json(sock, {"type": "video_register", "pc_id": self.pc_id})
            with self.conn_lock:
                self.video_sock = sock
            return True
        except OSError:
            return False

    def _connect_broadcast_video(self, generation: int) -> bool:
        if not self.pc_id:
            self._broadcast_diag("receiver_connect_skipped_no_pc_id")
            return False
        self._broadcast_diag("receiver_connect_enter", receiver_state="CONNECTING")
        sock: Optional[socket.socket] = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.teacher_ip, NETWORK.video_port))
            send_json(sock, {"type": "video_register", "pc_id": self.pc_id, "role": "broadcast_downlink"})
            self._broadcast_diag("receiver_registration_sent", receiver_state="REGISTERED", local_socket_id=id(sock))
            # Keep the active/generation check and publication ordered with Stop.
            # Stop holds state_lock while invalidating, then closes the published
            # socket; it therefore cannot leave a post-stop socket installed.
            with self.state_lock:
                if not self.broadcast_active or self.broadcast_generation != generation:
                    sock.close()
                    return False
                with self.conn_lock:
                    existing = self.broadcast_sock
                    self.broadcast_sock = sock
            if existing is not None and existing is not sock:
                try:
                    existing.close()
                except OSError:
                    pass
            return True
        except OSError as exc:
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
            self._broadcast_diag("receiver_connect_failed", receiver_state="RETRYING", reason=str(exc))
            return False

    def _connect_broadcast_audio(self) -> bool:
        if not self.pc_id:
            return False
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.teacher_ip, NETWORK.video_port))
            send_json(sock, {"type": "video_register", "pc_id": self.pc_id, "role": "broadcast_audio_downlink"})
            with self.conn_lock:
                existing = self.broadcast_audio_sock
                self.broadcast_audio_sock = sock
            if existing is not None and existing is not sock:
                try:
                    existing.close()
                except OSError:
                    pass
            return True
        except OSError:
            return False

    # def _close_broadcast_socket(self) -> None:
    #     with self.conn_lock:
    #         sock = self.broadcast_sock
    #         self.broadcast_sock = None
    #     if sock is not None:
    #         try:
    #             sock.close()
    #         except OSError:
    #             pass
    def _close_broadcast_socket(self, expected_sock: Optional[socket.socket] = None) -> None:
        """Close the current video downlink only when this caller owns it.

        A receiver blocked in recv_frame(old_sock) can wake after a new session
        has installed new_sock.  Passing old_sock prevents that stale receiver
        from clearing or closing new_sock.
        """
        with self.conn_lock:
            sock = self.broadcast_sock
            stale = expected_sock is not None and sock is not expected_sock
            if not stale:
                self.broadcast_sock = None

        if stale:
            self._broadcast_diag(
                "receiver_socket_close_skipped_stale",
                local_socket_id=id(expected_sock),
                current_socket_id=id(sock) if sock else None,
            )
            return

        self._broadcast_diag("receiver_socket_close_begin", local_socket_id=id(sock) if sock else None)

        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

            try:
                sock.close()
            except OSError:
                pass
        self._broadcast_diag("receiver_socket_close_complete", local_socket_id=id(sock) if sock else None)

    # def _close_broadcast_audio_socket(self) -> None:
    #     with self.conn_lock:
    #         sock = self.broadcast_audio_sock
    #         self.broadcast_audio_sock = None
    #     if sock is not None:
    #         try:
    #             sock.close()
    #         except OSError:
    #             pass
    def _close_broadcast_audio_socket(self) -> None:
        with self.conn_lock:
            sock = self.broadcast_audio_sock
            self.broadcast_audio_sock = None

        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

            try:
                sock.close()
            except OSError:
                pass

    def _cleanup_sockets(self) -> None:
        self._broadcast_diag("cleanup_sockets_begin")
        with self.state_lock:
            # Connection cleanup is another session boundary.  Invalidate queued
            # frames before their socket is detached.
            self.broadcast_generation += 1
            generation = self.broadcast_generation
            self.broadcast_active = False
        self.broadcast_diag_session = generation
        self.overlay.set_broadcast_generation(generation)
        with self.conn_lock:
            control_sock = self.control_sock
            video_sock = self.video_sock
            broadcast_sock = self.broadcast_sock
            broadcast_audio_sock = self.broadcast_audio_sock
            control_file = self.control_file
            self.control_sock = None
            self.video_sock = None
            self.broadcast_sock = None
            self.broadcast_audio_sock = None
            self.control_file = None
        if control_file is not None:
            try:
                control_file.close()
            except OSError:
                pass
        for sock in (control_sock, video_sock, broadcast_sock, broadcast_audio_sock):
            if sock:
                try:
                    sock.close()
                except OSError:
                    pass
        self._broadcast_diag("cleanup_sockets_complete", closed_broadcast_socket_id=id(broadcast_sock) if broadcast_sock else None)

    def _reconnect_loop(self) -> None:
        while True:
            try:
                snapshot = self._state_snapshot()
                if snapshot["connected"] or snapshot["connecting"]:
                    time.sleep(0.2)
                    continue
                self._update_state(connecting=True)
                self._show_connection_wait_ui()
                ok_control = self._connect_control()
                ok_video = self._connect_video() if ok_control else False
                is_connected = bool(ok_control and ok_video)
                self._update_state(connected=is_connected, connecting=False)
                if not is_connected:
                    self._show_connection_wait_ui()
                    self._cleanup_sockets()
                    sleep_time = self.dynamic_reconnect_interval_s + random.uniform(0, 0.4)
                    time.sleep(sleep_time)
            except Exception:
                self._update_state(connecting=False, connected=False)
                self._show_connection_wait_ui()
                self._cleanup_sockets()
                sleep_time = self.dynamic_reconnect_interval_s + random.uniform(0, 0.4)
                time.sleep(sleep_time)

    def _resource_snapshot(self) -> dict:
        try:
            cpu = float(psutil.cpu_percent(interval=None))
        except Exception:
            cpu = 0.0
        try:
            ram = float(psutil.virtual_memory().percent)
        except Exception:
            ram = 0.0
        try:
            disk = float(psutil.disk_usage("/").percent)
        except Exception:
            try:
                disk = float(psutil.disk_usage("C:\\").percent)
            except Exception:
                disk = 0.0
        uptime_s = int(max(0, time.time() - self.boot_ts))
        return {
            "cpu_percent": round(cpu, 2),
            "ram_percent": round(ram, 2),
            "disk_percent": round(disk, 2),
            "uptime_s": uptime_s,
        }

    def _heartbeat_loop(self) -> None:
        while True:
            try:
                with self.conn_lock:
                    control_sock = self.control_sock
                snapshot = self._state_snapshot()
                if snapshot["connected"] and control_sock and self.pc_id:
                    heartbeat = {"type": "heartbeat", "pc_id": self.pc_id}
                    heartbeat.update(self._resource_snapshot())
                    ok = self._safe_send_json(control_sock, heartbeat)
                    if not ok:
                        self._update_state(connected=False)
                        self._cleanup_sockets()
                time.sleep(RUNTIME.heartbeat_interval_s)
            except Exception:
                self._update_state(connected=False)
                self._cleanup_sockets()
                time.sleep(RUNTIME.heartbeat_interval_s)

    def _control_loop(self) -> None:
        while True:
            try:
                with self.conn_lock:
                    file_obj = self.control_file
                if not file_obj:
                    time.sleep(0.2)
                    continue
                msg = recv_json_line(file_obj)
                if msg is None:
                    self._update_state(connected=False)
                    self._cleanup_sockets()
                    continue
                msg_type = msg.get("type")
                if msg_type == "auth_ack":
                    self.auth_response_queue.put(msg)
                    action = str(msg.get("action", "")).strip().lower()
                    ok = bool(msg.get("ok"))
                    if action == "login" and ok and isinstance(msg.get("user"), dict):
                        self._update_state(
                            current_user=msg.get("user"),
                            signout_lock_active=False,
                            temporary_lock_active=False,
                        )
                        max_session_s = msg.get("max_session_s")
                        server_ts = msg.get("server_ts")
                        if isinstance(server_ts, (int, float)):
                            self._clear_session_timer()
                            self._set_session_timer_from_max_session(max_session_s, float(server_ts))
                        self._restore_overlay_state()
                    continue
                if msg_type != "command":
                    continue
                if msg.get("command") in {"BROADCAST_START", "BROADCAST_STOP"}:
                    self._broadcast_diag("control_command_dispatch", command=msg.get("command"), cmd_id=msg.get("cmd_id"))
                ack, post_action = self._execute_command(msg, from_udp=False)
                self._send_ack(ack)
                if post_action is not None:
                    post_action()
            except TimeoutError:
                continue
            except OSError:
                self._update_state(connected=False)
                self._cleanup_sockets()
            except Exception:
                self._update_state(connected=False)
                self._cleanup_sockets()
                time.sleep(0.2)

    def _video_loop(self) -> None:
        while True:
            try:
                with mss.mss() as sct:
                    monitor = sct.monitors[1]
                    while True:
                        with self.conn_lock:
                            video_sock = self.video_sock
                        if (not self._state_snapshot()["connected"]) or (not video_sock):
                            time.sleep(0.2)
                            continue
                        shot = sct.grab(monitor)
                        frame = cv2.cvtColor(numpy.array(shot), cv2.COLOR_BGRA2BGR)
                        if frame.shape[1] != self.stream_width or frame.shape[0] != self.stream_height:
                            frame = cv2.resize(frame, (self.stream_width, self.stream_height))
                        ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                        if ok:
                            try:
                                send_frame(video_sock, jpeg.tobytes())
                            except (OSError, AttributeError):
                                self._update_state(connected=False)
                                self._cleanup_sockets()
                        time.sleep(1 / max(1, self.stream_fps))
            except Exception:
                self._update_state(connected=False)
                self._cleanup_sockets()
                time.sleep(0.2)

    def _broadcast_receiver_loop(self) -> None:
        self._broadcast_diag("receiver_worker_started", receiver_state="IDLE")
        last_idle = None
        while True:
            sock: Optional[socket.socket] = None
            try:
                snapshot = self._state_snapshot()
                if not snapshot["broadcast_active"]:
                    if last_idle != snapshot["broadcast_active"]:
                        self._broadcast_diag("receiver_idle", receiver_state="IDLE")
                        last_idle = snapshot["broadcast_active"]
                    time.sleep(0.2)
                    continue

                last_idle = snapshot["broadcast_active"]

                with self.conn_lock:
                    sock = self.broadcast_sock
                if sock is None:
                    if (not snapshot["connected"]) or (not self.pc_id):
                        time.sleep(0.2)
                        continue
                    if not self._connect_broadcast_video(snapshot["broadcast_generation"]):
                        time.sleep(1)
                    continue

                recv_started = time.monotonic()
                recv_count_before = self.broadcast_diag_counters["frames_received"]
                log_recv_boundary = recv_count_before == 0 or (recv_count_before + 1) % 100 == 0
                if log_recv_boundary:
                    self._broadcast_diag("receiver_recv_enter", receiver_state="WAITING_FOR_FRAME", local_socket_id=id(sock), field_matches_local=(self.broadcast_sock is sock))
                frame_data = recv_frame(sock)
                recv_duration = round(time.monotonic() - recv_started, 6)
                if frame_data is None or log_recv_boundary:
                    self._broadcast_diag("receiver_recv_return", receiver_state="EOF" if frame_data is None else "FRAME_RECEIVED", local_socket_id=id(sock), field_matches_local=(self.broadcast_sock is sock), recv_duration_s=recv_duration, frame_size=len(frame_data) if frame_data else None)
                if frame_data is None:
                    self._close_broadcast_socket(sock)
                    time.sleep(0.2)
                    continue
                # recv_frame may have been unblocked by Stop, followed by a new
                # Start.  Do not decode, enqueue, or clean up a newer session.
                current = self._state_snapshot()
                with self.conn_lock:
                    still_owns_socket = self.broadcast_sock is sock
                if (
                    not current["broadcast_active"]
                    or current["broadcast_generation"] != snapshot["broadcast_generation"]
                    or not still_owns_socket
                ):
                    self._broadcast_diag(
                        "receiver_frame_discarded_stale",
                        local_socket_id=id(sock),
                        receiver_generation=snapshot["broadcast_generation"],
                        current_generation=current["broadcast_generation"],
                    )
                    continue
                self.broadcast_diag_counters["frames_received"] += 1
                if self.broadcast_diag_first_frame_mono is None:
                    self.broadcast_diag_first_frame_mono = time.monotonic()
                    self._broadcast_diag("receiver_first_frame_received", receiver_state="FRAME_RECEIVED")
                np_buf = numpy.frombuffer(frame_data, dtype=numpy.uint8)
                frame = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
                if frame is None:
                    self._broadcast_diag("receiver_decode_failed", receiver_state="FRAME_RECEIVED")
                    continue
                self.broadcast_diag_counters["frames_decoded"] += 1
                if self.broadcast_diag_counters["frames_decoded"] == 1:
                    self._broadcast_diag("receiver_first_frame_decoded", receiver_state="FRAME_RECEIVED")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Decoding is not instantaneous; validate again immediately
                # before handing the image to the asynchronous Tk queue.
                current = self._state_snapshot()
                with self.conn_lock:
                    still_owns_socket = self.broadcast_sock is sock
                if (
                    not current["broadcast_active"]
                    or current["broadcast_generation"] != snapshot["broadcast_generation"]
                    or not still_owns_socket
                ):
                    self._broadcast_diag(
                        "receiver_decoded_frame_discarded_stale",
                        local_socket_id=id(sock),
                        receiver_generation=snapshot["broadcast_generation"],
                        current_generation=current["broadcast_generation"],
                    )
                    continue
                queued = self.overlay.show_broadcast_frame_async(
                    Image.fromarray(rgb), snapshot["broadcast_generation"]
                )
                if queued:
                    self.broadcast_diag_counters["frames_enqueued"] += 1
                if self.broadcast_diag_counters["frames_enqueued"] == 1:
                    self._broadcast_diag("receiver_first_frame_enqueued", receiver_state="FRAME_RECEIVED", queued=queued)
            except OSError as exc:
                self._broadcast_diag("receiver_socket_error", receiver_state="SOCKET_ERROR", reason=str(exc))
                # sock is intentionally local to the last recv iteration.  It
                # may be stale, so cleanup must retain its identity.
                self._close_broadcast_socket(locals().get("sock"))
                time.sleep(0.5)
            except Exception as exc:
                self._broadcast_diag("receiver_exception", receiver_state="RETRYING", reason=str(exc))
                self._close_broadcast_socket(locals().get("sock"))
                time.sleep(0.5)

    def _broadcast_audio_receiver_loop(self) -> None:
        while True:
            try:
                snapshot = self._state_snapshot()
                if not snapshot["broadcast_active"]:
                    self._close_broadcast_audio_socket()
                    time.sleep(0.2)
                    continue
                if sd is None:
                    if not self._broadcast_audio_missing_warned:
                        print("[Broadcast Audio] sounddevice is not installed.")
                        self._broadcast_audio_missing_warned = True
                    time.sleep(2.0)
                    continue
                self._broadcast_audio_missing_warned = False
                try:
                    with sd.OutputStream(
                        samplerate=AUDIO_SAMPLE_RATE,
                        channels=AUDIO_CHANNELS,
                        dtype=AUDIO_DTYPE,
                        blocksize=AUDIO_BLOCK_SIZE,
                    ) as stream:
                        self._broadcast_audio_device_warned = False
                        while True:
                            snapshot = self._state_snapshot()
                            if not snapshot["broadcast_active"]:
                                self._close_broadcast_audio_socket()
                                break
                            with self.conn_lock:
                                sock = self.broadcast_audio_sock
                            if sock is None:
                                if (not snapshot["connected"]) or (not self.pc_id):
                                    time.sleep(0.2)
                                    continue
                                if not self._connect_broadcast_audio():
                                    time.sleep(1.0)
                                continue
                            audio_data = recv_frame(sock)
                            if audio_data is None:
                                self._close_broadcast_audio_socket()
                                time.sleep(0.2)
                                continue
                            if len(audio_data) % numpy.dtype(numpy.int16).itemsize != 0:
                                continue
                            samples = numpy.frombuffer(audio_data, dtype=numpy.int16)
                            if samples.size == 0:
                                continue
                            stream.write(samples.reshape(-1, AUDIO_CHANNELS))
                except Exception as exc:
                    if not self._broadcast_audio_device_warned:
                        print(f"[Broadcast Audio] {exc}")
                        self._broadcast_audio_device_warned = True
                    self._close_broadcast_audio_socket()
                    time.sleep(1.0)
            except OSError:
                self._close_broadcast_audio_socket()
                time.sleep(0.5)
            except Exception:
                self._close_broadcast_audio_socket()
                time.sleep(0.5)

    def _on_timer_warning(self, timer_id: str, remaining_ms: int) -> None:
        if not self.enable_timer_near_limit_notify:
            return
        print("Your timing is nearing the limit")
        with self.conn_lock:
            control_sock = self.control_sock
        if control_sock and self.pc_id:
            self._safe_send_json(control_sock, {
                "type": "ack",
                "pc_id": self.pc_id,
                "command": "TIMER_WARNING",
                "timer_id": timer_id,
                "result": "applied",
                "reason": "",
            })

    def send_session_message(self, text: str) -> bool:
        snapshot = self._state_snapshot()
        if not self.enable_session_messaging or (not snapshot["connected"]):
            return False
        message = str(text or "").strip()
        if not message:
            return False
        with self.conn_lock:
            sock = self.control_sock
        if not sock or not self.pc_id:
            return False
        payload = {
            "type": "student_session_message",
            "pc_id": self.pc_id,
            "text": message,
            "timestamp": time.time(),
        }
        return self._safe_send_json(sock, payload)

    def request_extension(self, requested_extra_ms: int) -> bool:
        snapshot = self._state_snapshot()
        if not self.enable_extension_requests or (not snapshot["connected"]):
            return False
        with self.conn_lock:
            sock = self.control_sock
        if not sock:
            return False
        payload = {
            "type": "student_extension_request",
            "requested_extra_ms": max(60000, int(requested_extra_ms)),
        }
        return self._safe_send_json(sock, payload)

    def _on_timer_expired(self, timer_id: str) -> None:
        if timer_id == self.session_timer_id:
            self._update_state(current_user=None, signout_lock_active=True, temporary_lock_active=False)
            self._restore_overlay_state()
        with self.conn_lock:
            control_sock = self.control_sock
        if control_sock and self.pc_id:
            self._safe_send_json(control_sock, {
                "type": "ack",
                "pc_id": self.pc_id,
                "command": "TIMER_EXPIRED",
                "timer_id": timer_id,
                "result": "applied",
                "reason": "",
            })

    def run(self) -> None:
        threading.Thread(target=self._reconnect_loop, daemon=True).start()
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        threading.Thread(target=self._control_loop, daemon=True).start()
        threading.Thread(target=self._udp_fallback_loop, daemon=True).start()
        threading.Thread(target=self._video_loop, daemon=True).start()
        threading.Thread(target=self._broadcast_receiver_loop, daemon=True).start()
        threading.Thread(target=self._broadcast_audio_receiver_loop, daemon=True).start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self._cleanup_sockets()
            self.overlay.set_state(OverlayState.HIDDEN)


def main() -> None:
    settings_store = StudentSettingsStore(default_teacher_host=NETWORK.teacher_connect_host)
    settings = settings_store.load()
    client = StudentDeployClient(settings.teacher_host, settings_store=settings_store)
    client = StudentDeployClient(NETWORK.teacher_host)
    try:
        client.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()







