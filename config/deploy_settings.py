from dataclasses import dataclass


@dataclass(frozen=True)
class NetworkConfig:
    teacher_bind_host: str = "0.0.0.0"
    teacher_connect_host: str = "192.168.1.157" ##110.236  main 1.127

    @property
    def teacher_host(self) -> str:
        return self.teacher_connect_host
    control_port: int = 9201
    video_port: int = 9200
    sensor_port: int = 9202
    command_fallback_port: int = 9203
    screen_share_video_port: int = 9204
    screen_share_audio_port: int = 9205


@dataclass(frozen=True)
class RuntimeConfig:
    max_fps: int = 5
    heartbeat_interval_s: int = 5
    heartbeat_timeout_s: int = 10
    reconnect_interval_s: int = 3
    # frame_width: int = 640
    # frame_height: int = 360
    frame_width: int = 1280
    frame_height: int = 720
    frame_queue_max: int = 5
    sensor_timeout_s: int = 10
    timer_tick_s: float = 0.5
    session_disconnect_grace_s: int = 30
    screen_share_width: int = 1280
    screen_share_height: int = 720
    screen_share_fps: int = 15
    screen_share_jpeg_quality: int = 78
    screen_share_audio_rate: int = 44100
    screen_share_audio_channels: int = 2
    screen_share_audio_chunk_frames: int = 1024


NETWORK = NetworkConfig()
RUNTIME = RuntimeConfig()

# from dataclasses import dataclass


# @dataclass(frozen=True)
# class NetworkConfig:
#     teacher_host: str = "0.0.0.0"
#     control_port: int = 9201
#     video_port: int = 9200
#     sensor_port: int = 9202


# @dataclass(frozen=True)
# class RuntimeConfig:
#     max_fps: int = 10
#     heartbeat_interval_s: int = 5
#     heartbeat_timeout_s: int = 10
#     reconnect_interval_s: int = 3
#     frame_width: int = 1280
#     frame_height: int = 720
#     frame_queue_max: int = 5
#     sensor_timeout_s: int = 10
#     timer_tick_s: float = 0.5


# STREAM_PROFILES: dict[str, tuple[int, int]] = {
#     "360p": (640, 360),
#     "720p": (1280, 720),
#     "1080p": (1920, 1080),
# }

# DEFAULT_STREAM_PROFILE = "720p"


# NETWORK = NetworkConfig()
# RUNTIME = RuntimeConfig()


# .\venv\Scripts\Activate.ps1   
# py -m venv venv  
# pip install -r requirements.txt     
# python.exe -m pip install --upgrade pip    