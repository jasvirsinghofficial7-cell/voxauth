"""
database.py — SQLite Database Layer for Voice Authorization App
Handles: User registration/login, per-user audio & voiceprint storage
"""

import sqlite3
import hashlib
import io
import numpy as np
import scipy.io.wavfile as wavfile
import os

DB_PATH = "voice_auth.db"


# ─────────────────────────────────────────────
# 1.  Core DB Connection
# ─────────────────────────────────────────────
def get_connection():
    """Return a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ─────────────────────────────────────────────
# 2.  Initialization
# ─────────────────────────────────────────────
def init_db():
    """Create the global 'users' table if it doesn't already exist."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    NOT NULL UNIQUE,
            password_hash TEXT  NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
# 3.  Per-User Table Helpers
# ─────────────────────────────────────────────
def _safe_table_name(username: str) -> str:
    """Sanitize username to use as a table name prefix."""
    return "".join(c if c.isalnum() else "_" for c in username.lower())


def _create_user_tables(username: str):
    """
    Create two per-user tables:
      • audio_<username>      — stores raw & processed WAV audio blobs
      • voiceprint_<username> — stores the 192-dim ECAPA embedding
    """
    safe = _safe_table_name(username)
    conn = get_connection()
    cur = conn.cursor()

    # Audio table: both 'raw' (enrollment recording) and 'processed' (normalized)
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS audio_{safe} (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            audio_type  TEXT    NOT NULL,          -- 'raw_enrollment' | 'normalized_enrollment' | 'verification'
            sample_rate INTEGER NOT NULL DEFAULT 16000,
            audio_blob  BLOB    NOT NULL,           -- WAV file as bytes
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Voiceprint table: 192-dim float32 numpy array stored as BLOB
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS voiceprint_{safe} (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            embedding   BLOB    NOT NULL,           -- numpy array serialized with np.save
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
# 4.  Authentication Functions
# ─────────────────────────────────────────────
def _hash_password(password: str) -> str:
    """SHA-256 hash a password string."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def register_user(username: str, password: str) -> tuple[bool, str]:
    """
    Register a new user.
    Returns (True, "success") on success, or (False, error_message) on failure.
    """
    if not username.strip() or not password.strip():
        return False, "Username and password cannot be empty."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username.strip().lower(), _hash_password(password))
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return False, f"Username '{username}' is already taken. Please choose another."
    conn.close()

    # Create this user's dedicated audio & voiceprint tables
    _create_user_tables(username.strip().lower())
    return True, "Registration successful!"


def authenticate_user(username: str, password: str) -> tuple[bool, str]:
    """
    Verify credentials.
    Returns (True, "success") or (False, error_message).
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT password_hash FROM users WHERE username = ?",
        (username.strip().lower(),)
    )
    row = cur.fetchone()
    conn.close()

    if row is None:
        return False, "User not found. Please register first."
    if row["password_hash"] != _hash_password(password):
        return False, "Incorrect password. Please try again."
    return True, "Login successful!"


# ─────────────────────────────────────────────
# 5.  Audio Storage
# ─────────────────────────────────────────────
def _numpy_to_wav_bytes(audio_np: np.ndarray, sample_rate: int) -> bytes:
    """Convert a float32 numpy array to WAV bytes (in-memory)."""
    # Scale to int16 for WAV format
    audio_int16 = (audio_np * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, audio_int16)
    return buf.getvalue()


def save_audio_to_db(username: str, audio_type: str, audio_np: np.ndarray, sample_rate: int = 16000):
    """
    Save a numpy audio array to the user's audio table.
    audio_type: 'raw_enrollment' | 'normalized_enrollment' | 'verification'
    """
    safe = _safe_table_name(username)
    wav_bytes = _numpy_to_wav_bytes(audio_np, sample_rate)

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        f"INSERT INTO audio_{safe} (audio_type, sample_rate, audio_blob) VALUES (?, ?, ?)",
        (audio_type, sample_rate, wav_bytes)
    )
    conn.commit()
    conn.close()


def load_audio_from_db(username: str, audio_type: str = None) -> list[dict]:
    """
    Load audio records for a user. Optionally filter by audio_type.
    Returns list of dicts with keys: id, audio_type, sample_rate, audio_blob, recorded_at
    """
    safe = _safe_table_name(username)
    conn = get_connection()
    cur = conn.cursor()
    if audio_type:
        cur.execute(f"SELECT * FROM audio_{safe} WHERE audio_type = ? ORDER BY recorded_at DESC", (audio_type,))
    else:
        cur.execute(f"SELECT * FROM audio_{safe} ORDER BY recorded_at DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# ─────────────────────────────────────────────
# 6.  Voiceprint Storage
# ─────────────────────────────────────────────
def save_voiceprint_to_db(username: str, embedding_np: np.ndarray):
    """
    Save (or replace) a user's voiceprint embedding in the DB.
    Old embeddings are kept for history; latest is used for verification.
    """
    safe = _safe_table_name(username)
    buf = io.BytesIO()
    np.save(buf, embedding_np)
    blob = buf.getvalue()

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(f"INSERT INTO voiceprint_{safe} (embedding) VALUES (?)", (blob,))
    conn.commit()
    conn.close()


def load_voiceprint_from_db(username: str) -> np.ndarray | None:
    """
    Load the most recent voiceprint embedding for a user.
    Returns numpy array or None if not found.
    """
    safe = _safe_table_name(username)
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT embedding FROM voiceprint_{safe} ORDER BY created_at DESC LIMIT 1")
        row = cur.fetchone()
    except sqlite3.OperationalError:
        conn.close()
        return None
    conn.close()

    if row is None:
        return None

    buf = io.BytesIO(row["embedding"])
    return np.load(buf)


# ─────────────────────────────────────────────
# 7.  Utility
# ─────────────────────────────────────────────
def get_all_users() -> list[str]:
    """Return list of all registered usernames."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT username FROM users ORDER BY created_at DESC")
    rows = [r["username"] for r in cur.fetchall()]
    conn.close()
    return rows


def get_audio_count(username: str) -> dict:
    """Return count of audio recordings per type for a user."""
    safe = _safe_table_name(username)
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT audio_type, COUNT(*) as cnt FROM audio_{safe} GROUP BY audio_type")
        result = {row["audio_type"]: row["cnt"] for row in cur.fetchall()}
    except sqlite3.OperationalError:
        result = {}
    conn.close()
    return result
