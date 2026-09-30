"""Persistent store for batch cycles and the violations they produce.

The dashboard's recap tiles need history, and `runs/_latest/` only ever holds
the newest cycle. SQLite (stdlib, no new dependency) rather than an append-only
log because an operator's verification verdict is an *update* to an existing
event, not a new fact.

Deduplication across cycles is deliberately absent: every cycle analyses a
freshly captured clip, so track id 2 in one cycle and track id 2 in the next are
different vehicles. Each cycle's events are distinct occurrences.

Times are stored as UTC ISO strings. "Today" is computed in WIB, because a
recap for a Jakarta agency that rolls over at 07:00 local time would be wrong.
Jakarta is UTC+7 all year with no DST, so a fixed offset is exact and avoids
depending on a tz database inside the container.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .settings import settings

WIB = timezone(timedelta(hours=7), "WIB")
DB_PATH = settings.runs_dir / "lalin.db"

_lock = threading.Lock()

# Which agency may act on a violation. Dishub's authority is administrative
# (parking, public transport, goods vehicles); moving violations are Polri's,
# so those are recorded as referrals rather than actions.
AUTHORITY = {
    "parkir_liar": "dishub",
    "ngetem": "dishub",
    "jalur_sepeda": "dishub",
    "angkutan_barang": "dishub",
    "busway": "rujuk",
    "lawan_arah": "rujuk",
    "langgar_marka": "rujuk",
}

LABEL = {
    "parkir_liar": "Parkir di badan jalan",
    "ngetem": "Angkutan umum ngetem",
    "jalur_sepeda": "Kendaraan bermotor di jalur sepeda",
    "angkutan_barang": "Angkutan barang pada jam larangan",
    "busway": "Okupansi lajur TransJakarta",
    "lawan_arah": "Kendaraan lawan arah",
    "langgar_marka": "Pelanggaran marka lajur",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS cycles (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    camera        TEXT NOT NULL,
    scenario      TEXT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT NOT NULL,
    ok            INTEGER NOT NULL,
    frames        INTEGER DEFAULT 0,
    unique_tracks INTEGER DEFAULT 0,
    elapsed_sec   REAL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id      INTEGER REFERENCES cycles(id),
    camera        TEXT NOT NULL,
    kind          TEXT NOT NULL,
    authority     TEXT NOT NULL,
    track_id      INTEGER,
    first_sec     REAL,
    duration_sec  REAL,
    threshold_sec INTEGER,
    detected_at   TEXT NOT NULL,
    verified      INTEGER NOT NULL DEFAULT 0,
    verdict       TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_detected ON events(detected_at);
CREATE INDEX IF NOT EXISTS idx_cycles_finished ON cycles(finished_at);
"""


# ---------------------------------------------------------------------------
# Lapisan basis data dua dialek.
#
# SQL di modul ini dan di aduan.py ditulis dalam SATU bentuk yang sah di
# SQLite >= 3.35 dan PostgreSQL: placeholder `?`, TRUE/FALSE untuk boolean,
# INSERT ... RETURNING id, ON CONFLICT ... DO NOTHING. Lapisan ini hanya
# mengurus dua hal yang memang berbeda:
#   1. placeholder: `?` diterjemahkan ke `%s` untuk psycopg;
#   2. nilai keluaran: PostgreSQL mengembalikan datetime, bool, dan dict/list
#      (JSONB), sedangkan sisa aplikasi - dan dashboard - dibangun di atas
#      bentuk SQLite: teks ISO-8601 UTC, 0/1, dan teks JSON. Nilai PostgreSQL
#      disamakan ke bentuk itu, supaya tidak ada satu pun pemanggil yang perlu
#      tahu dialek mana yang sedang dipakai.
# Parameter berupa teks ISO dan teks JSON dikirim apa adanya: psycopg mengirim
# str sebagai tipe "unknown", sehingga PostgreSQL sendiri yang menafsirkannya
# sebagai TIMESTAMPTZ atau JSONB sesuai kolom tujuannya.
# ---------------------------------------------------------------------------
DIALEK = "pg" if settings.db_host else "sqlite"


class Baris(dict):
    """Baris hasil query yang bisa dibaca per nama (r["id"]) maupun per posisi
    (r[0]) - seperti sqlite3.Row, yang dipakai di seluruh modul ini."""
    __slots__ = ("_urut",)

    def __init__(self, kolom, nilai):
        super().__init__(zip(kolom, nilai))
        self._urut = list(nilai)

    def __getitem__(self, k):
        return self._urut[k] if isinstance(k, int) else super().__getitem__(k)


def _normal(v):
    if isinstance(v, datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc).isoformat(timespec="seconds")
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return v


class _Kursor:
    def __init__(self, cur):
        self._cur = cur
        self.rowcount = cur.rowcount

    def _baris(self, r):
        if r is None:
            return None
        kolom = [d[0] for d in self._cur.description]
        return Baris(kolom, [_normal(v) for v in r])

    def fetchone(self):
        return self._baris(self._cur.fetchone())

    def fetchall(self):
        return [self._baris(r) for r in self._cur.fetchall()]

    def __iter__(self):
        return iter(self.fetchall())


class _KoneksiPG:
    """Pembungkus psycopg dengan antarmuka yang dipakai modul ini:
    execute() langsung pada koneksi, dan `with` = commit lalu tutup."""

    def __init__(self):
        import psycopg
        pw = os.environ.get("LALIN_APP_PASSWORD", "")
        self._c = psycopg.connect(host=settings.db_host, port=settings.db_port,
                                  dbname=settings.db_name, user=settings.db_user,
                                  password=pw, connect_timeout=10,
                                  options="-c search_path=lalin")

    def execute(self, sql, params=()):
        cur = self._c.cursor()
        cur.execute(re.sub(r"\?", "%s", sql), tuple(params))
        return _Kursor(cur)

    def __enter__(self):
        return self

    def __exit__(self, jenis, *_):
        try:
            if jenis is None:
                self._c.commit()
            else:
                self._c.rollback()
        finally:
            self._c.close()
        return False


def _connect():
    if DIALEK == "pg":
        return _KoneksiPG()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    # Wajib per koneksi: tanpa ini SQLite mengabaikan REFERENCES dan
    # ON DELETE CASCADE pada tabel aduan.
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


SQL_DIR = Path(__file__).parent / "sql"

# Kolom yang ditambahkan setelah tabel events pertama kali dibuat. SQLite tidak
# punya ADD COLUMN IF NOT EXISTS, jadi keberadaannya diperiksa dulu. Tanpa
# ketiganya, keputusan verifikasi tidak bisa diaudit ketika dipakai sebagai
# bukti untuk aduan masyarakat: siapa yang memutus, kapan, dan mengapa.
MIGRASI_EVENTS = (
    ("verified_at", "TEXT"),
    ("verified_by", "TEXT"),
    ("verify_note", "TEXT"),
)


def init() -> None:
    if DIALEK == "pg":
        # Skema PostgreSQL dikelola sebagai DDL tersendiri (docker/pg-init/,
        # sumbernya docs/ddl/skema_postgres.sql), dibuat sebagai lalin_app saat
        # volume database pertama kali dibuat. Di sini hanya diperiksa.
        with _lock, _connect() as conn:
            ada = {r[0] for r in conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'lalin'")}
        kurang = {"cycles", "events", "aduan", "aduan_kecocokan", "aduan_riwayat",
                  "kamera_lokasi"} - ada
        if kurang:
            raise RuntimeError(f"tabel PostgreSQL belum ada: {sorted(kurang)}")
        return
    with _lock, _connect() as conn:
        conn.executescript(SCHEMA)
        ada = {r["name"] for r in conn.execute("PRAGMA table_info(events)")}
        for kolom, tipe in MIGRASI_EVENTS:
            if kolom not in ada:
                conn.execute(f"ALTER TABLE events ADD COLUMN {kolom} {tipe}")
        # DDL aduan disimpan sebagai berkas .sql tersendiri: itulah dokumen
        # skema yang diserahkan ke tim DB, dan menjalankannya dari sini menjamin
        # dokumen dan basis data tidak pernah berselisih.
        conn.executescript((SQL_DIR / "aduan.sql").read_text(encoding="utf-8"))


def _wib_day_bounds(day: datetime | None = None) -> tuple[str, str]:
    """UTC ISO bounds of one WIB calendar day."""
    now = (day or datetime.now(WIB)).astimezone(WIB)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return (start.astimezone(timezone.utc).isoformat(),
            end.astimezone(timezone.utc).isoformat())


def record_cycle(record: dict[str, Any]) -> int | None:
    """Persist one camera's analysis. Returns the cycle id."""
    results = record.get("results") or {}
    with _lock, _connect() as conn:
        cur = conn.execute(
            "INSERT INTO cycles (camera, scenario, started_at, finished_at, ok,"
            " frames, unique_tracks, elapsed_sec) VALUES (?,?,?,?,?,?,?,?) RETURNING id",
            (record.get("key"), record.get("scenario"),
             record.get("started_at"), record.get("finished_at"),
             bool(record.get("ok")),
             int(results.get("frames") or 0),
             int(results.get("unique_tracks") or 0),
             float(record.get("elapsed_sec") or 0)))
        cycle_id = cur.fetchone()[0]

        for ev in results.get("events") or []:
            kind = ev.get("kind", "parkir_liar")
            cur_ev = conn.execute(
                "INSERT INTO events (cycle_id, camera, kind, authority,"
                " track_id, first_sec, duration_sec, threshold_sec, detected_at)"
                " VALUES (?,?,?,?,?,?,?,?,?) RETURNING id",
                (cycle_id, record.get("key"), kind,
                 AUTHORITY.get(kind, "dishub"),
                 ev.get("track_id"), ev.get("first_sec"),
                 ev.get("duration_sec"), ev.get("threshold_sec"),
                 record.get("finished_at")))
            # Dicatat balik ke kejadiannya: batch memakai ID ini untuk menamai
            # foto bukti, dan dashboard untuk menautkan kejadian ke fotonya.
            ev["event_id"] = cur_ev.fetchone()[0]
    return cycle_id


def verify(event_id: int, verdict: str, oleh: str = "petugas",
           catatan: str | None = None) -> bool:
    """Record an operator's decision. verdict: 'benar' | 'bukan'."""
    with _lock, _connect() as conn:
        cur = conn.execute(
            "UPDATE events SET verified = TRUE, verdict = ?, verified_at = ?,"
            " verified_by = ?, verify_note = ? WHERE id = ?",
            (verdict, datetime.now(timezone.utc).isoformat(timespec="seconds"),
             oleh, catatan, event_id))
        return cur.rowcount > 0


def recent_events(limit: int = 40, today_only: bool = True) -> list[dict]:
    """Temuan terbaru. Bawaannya DIBATASI hari WIB berjalan, sama dengan
    stats(): daftar di dashboard dan kartu rekap membaca dari sini, dan keduanya
    harus menghitung himpunan yang sama. Tanpa batas ini, lewat tengah malam
    daftar masih memperlihatkan temuan kemarin sementara kartu sudah nol."""
    sql = "SELECT * FROM events"
    params: list = []
    if today_only:
        start, end = _wib_day_bounds()
        sql += " WHERE detected_at >= ? AND detected_at < ?"
        params += [start, end]
    sql += " ORDER BY detected_at DESC, id DESC LIMIT ?"
    params.append(limit)
    with _lock, _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [{**dict(r), "label": LABEL.get(r["kind"], r["kind"])} for r in rows]


def list_events(page: int = 1, per_page: int = 20, status: str | None = None,
                camera: str | None = None, kind: str | None = None,
                authority: str | None = None, today_only: bool = False) -> dict:
    """Daftar berhalaman untuk halaman Antrean Verifikasi.

    `jumlah` dihitung dengan saringan yang SAMA kecuali status, supaya tab
    "Belum / Sudah / Semua" selalu menjumlah ke angka yang sama dengan yang
    sedang dilihat petugas - bukan ke total seluruh basis data.
    """
    page = max(1, int(page))
    per_page = min(100, max(5, int(per_page)))
    syarat: list[str] = []
    params: list = []
    if today_only:
        start, end = _wib_day_bounds()
        syarat.append("detected_at >= ? AND detected_at < ?")
        params += [start, end]
    for kolom, nilai in (("camera", camera), ("kind", kind), ("authority", authority)):
        if nilai:
            syarat.append(f"{kolom} = ?")
            params.append(nilai)
    dasar = (" WHERE " + " AND ".join(syarat)) if syarat else ""

    saring_status = ""
    if status == "belum":
        saring_status = " verified = FALSE"
    elif status == "sudah":
        saring_status = " verified = TRUE"
    where = dasar
    if saring_status:
        where = (dasar + " AND" + saring_status) if dasar else (" WHERE" + saring_status)

    with _lock, _connect() as conn:
        jumlah = conn.execute(
            "SELECT COUNT(*) semua, SUM(CASE WHEN verified = FALSE THEN 1 ELSE 0 END) belum,"
            " SUM(CASE WHEN verified = TRUE THEN 1 ELSE 0 END) sudah FROM events" + dasar,
            params).fetchone()
        total = conn.execute("SELECT COUNT(*) FROM events" + where, params).fetchone()[0]
        rows = conn.execute(
            "SELECT * FROM events" + where
            + " ORDER BY detected_at DESC, id DESC LIMIT ? OFFSET ?",
            params + [per_page, (page - 1) * per_page]).fetchall()
        kamera = [r[0] for r in conn.execute(
            "SELECT DISTINCT camera FROM events ORDER BY camera").fetchall()]
        jenis = [r[0] for r in conn.execute(
            "SELECT DISTINCT kind FROM events ORDER BY kind").fetchall()]
    return {
        "items": [{**dict(r), "label": LABEL.get(r["kind"], r["kind"])} for r in rows],
        "total": total, "page": page, "per_page": per_page,
        "pages": max(1, -(-total // per_page)),
        "jumlah": {"semua": jumlah["semua"] or 0, "belum": jumlah["belum"] or 0,
                   "sudah": jumlah["sudah"] or 0},
        "pilihan": {"kamera": kamera,
                    "jenis": [{"kind": j, "label": LABEL.get(j, j)} for j in jenis]},
    }


def get_event(event_id: int) -> dict | None:
    with _lock, _connect() as conn:
        r = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
    return {**dict(r), "label": LABEL.get(r["kind"], r["kind"])} if r else None


def stats_per_camera() -> list[dict]:
    """Rincian hari berjalan per kamera, memakai batas hari WIB yang sama
    dengan stats(). Jumlah kolom `tracks` di sini sama persis dengan
    `vehicles_tracked` pada stats(), supaya kartu dan rinciannya tidak pernah
    berselisih."""
    start, end = _wib_day_bounds()
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT camera,"
            "       COUNT(*) AS cycles,"
            "       SUM(CASE WHEN ok = TRUE THEN 1 ELSE 0 END) AS ok_cycles,"
            "       COALESCE(SUM(CASE WHEN ok = TRUE THEN unique_tracks ELSE 0 END), 0) AS tracks,"
            "       COALESCE(SUM(CASE WHEN ok = TRUE THEN frames ELSE 0 END), 0) AS frames,"
            "       MAX(finished_at) AS last_at"
            "  FROM cycles WHERE finished_at >= ? AND finished_at < ?"
            " GROUP BY camera", (start, end)).fetchall()
        ev = {r["camera"]: r["c"] for r in conn.execute(
            "SELECT camera, COUNT(*) c FROM events"
            " WHERE detected_at >= ? AND detected_at < ? GROUP BY camera",
            (start, end)).fetchall()}
    out = [{"camera": r["camera"], "cycles": r["cycles"],
            "ok_cycles": r["ok_cycles"] or 0, "tracks": int(r["tracks"] or 0),
            "frames": int(r["frames"] or 0), "last_at": r["last_at"],
            "events": ev.get(r["camera"], 0)} for r in rows]
    out.sort(key=lambda x: -x["tracks"])
    return out


def stats() -> dict:
    """Everything the recap tiles and the hourly chart need."""
    start, end = _wib_day_bounds()
    with _lock, _connect() as conn:
        total = conn.execute(
            "SELECT COUNT(*) c FROM events WHERE detected_at >= ? AND detected_at < ?",
            (start, end)).fetchone()["c"]
        by_auth = {r["authority"]: r["c"] for r in conn.execute(
            "SELECT authority, COUNT(*) c FROM events"
            " WHERE detected_at >= ? AND detected_at < ? GROUP BY authority",
            (start, end)).fetchall()}
        verified = conn.execute(
            "SELECT COUNT(*) c FROM events WHERE verified = TRUE"
            " AND detected_at >= ? AND detected_at < ?",
            (start, end)).fetchone()["c"]
        tracks = conn.execute(
            "SELECT COALESCE(SUM(unique_tracks), 0) s FROM cycles"
            " WHERE ok = TRUE AND finished_at >= ? AND finished_at < ?",
            (start, end)).fetchone()["s"]
        cycles = conn.execute(
            "SELECT COUNT(*) c FROM cycles WHERE finished_at >= ? AND finished_at < ?",
            (start, end)).fetchone()["c"]
        rows = conn.execute(
            "SELECT detected_at FROM events"
            " WHERE detected_at >= ? AND detected_at < ?", (start, end)).fetchall()

    hourly = {h: 0 for h in range(24)}
    for r in rows:
        try:
            hour = datetime.fromisoformat(r["detected_at"]).astimezone(WIB).hour
            hourly[hour] += 1
        except ValueError:
            pass

    return {
        "day_wib": datetime.now(WIB).strftime("%Y-%m-%d"),
        "events_today": total,
        "by_authority": {"dishub": by_auth.get("dishub", 0),
                         "rujuk": by_auth.get("rujuk", 0)},
        "verified": verified,
        "unverified": total - verified,
        "vehicles_tracked": int(tracks or 0),
        "cycles_today": cycles,
        "hourly": [{"hour": h, "count": c} for h, c in sorted(hourly.items())],
        "has_data": cycles > 0,
    }


init()
