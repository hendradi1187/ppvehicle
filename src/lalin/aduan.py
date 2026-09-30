"""Aduan masyarakat dan pencocokannya dengan kejadian terekam.

Skema: src/lalin/sql/aduan.sql. Modul ini sengaja tidak mengimpor `batch`:
data kamera (nama, area, wilayah) diteruskan oleh pemanggil, supaya tidak ada
impor melingkar dengan batch -> store.

Aturan kejujuran yang dipegang di sini:
  * "tidak ditemukan" hanya boleh dinyatakan bila kamera itu MEMANG terekam
    pada rentang waktu aduan. Sistem memantau lewat klip berkala, bukan 24 jam;
    tanpa rekaman, ketiadaan kejadian bukan bukti apa pun -> "tidak terpantau".
  * Kejadian yang sudah diputus petugas sebagai "bukan pelanggaran" tidak
    pernah dijadikan kandidat bukti.
  * Skor pencocokan menyimpan rinciannya, supaya bisa diaudit.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from . import store

# Kategori aduan -> jenis kejadian yang bisa membuktikannya. Kategori tanpa
# pasangan (kemacetan, lainnya) belum dideteksi otomatis; aduannya tetap
# tercatat, tetapi pencocokan menyatakannya terus terang.
KATEGORI_KE_JENIS: dict[str, tuple[str, ...]] = {
    "parkir_liar": ("parkir_liar",),
    "lawan_arah": ("lawan_arah",),
    "langgar_marka": ("langgar_marka",),
    # Zona "jalur sepeda" di Thamrin dan S. Parman 07 mencatat kendaraan yang
    # berhenti di jalur sepeda sebagai parkir_liar.
    "jalur_sepeda": ("jalur_sepeda", "parkir_liar"),
    "ngetem": ("ngetem", "parkir_liar"),
    "kemacetan": (),
    "lainnya": (),
}
LABEL_KATEGORI = {
    "parkir_liar": "Parkir liar", "lawan_arah": "Kendaraan lawan arah",
    "langgar_marka": "Pelanggaran marka", "jalur_sepeda": "Kendaraan di jalur sepeda",
    "ngetem": "Angkutan umum ngetem", "kemacetan": "Kemacetan", "lainnya": "Lainnya",
}
SUMBER = ("crm", "jaki", "telepon", "medsos", "petugas")
STATUS = ("baru", "ada_kandidat", "terbukti", "tidak_ditemukan",
          "tidak_terpantau", "di_luar_cakupan", "selesai")

JENDELA_MENIT = 30          # rentang waktu bawaan di sekitar waktu kejadian
BOBOT = {"jenis": 0.40, "kamera_ditunjuk": 0.30, "kamera_koordinat": 0.25,
         "kamera_teks": 0.15, "waktu": 0.20, "terverifikasi": 0.10}


def _kini() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _iso(teks: str) -> str:
    """Terima ISO dengan zona apa pun (atau tanpa zona = WIB), simpan UTC."""
    t = datetime.fromisoformat(str(teks).strip().replace("Z", "+00:00"))
    if t.tzinfo is None:
        t = t.replace(tzinfo=store.WIB)       # waktu tanpa zona dianggap WIB
    return t.astimezone(timezone.utc).isoformat(timespec="seconds")


def _jarak_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _kata(teks: str) -> set[str]:
    buang = {"jl", "jalan", "simpang", "jpo", "cctv", "raya", "dan", "di", "depan", "dekat"}
    return {k for k in re.findall(r"[a-z0-9]+", (teks or "").lower())
            if len(k) >= 4 and k not in buang}


def _riwayat(conn, aduan_id: int, dari: str | None, ke: str, oleh: str, catatan: str | None):
    conn.execute(
        "INSERT INTO aduan_riwayat (aduan_id, waktu, dari_status, ke_status, oleh, catatan)"
        " VALUES (?,?,?,?,?,?)", (aduan_id, _kini(), dari, ke, oleh, catatan))


# ------------------------------------------------------------------ tulis
def validasi(data: dict) -> dict:
    kategori = str(data.get("kategori") or "").strip()
    if kategori not in KATEGORI_KE_JENIS:
        raise ValueError(f"kategori tidak dikenal: {kategori!r}")
    sumber = str(data.get("sumber") or "crm").strip()
    if sumber not in SUMBER:
        raise ValueError(f"sumber tidak dikenal: {sumber!r}")
    lokasi = str(data.get("lokasi_teks") or "").strip()
    if not lokasi:
        raise ValueError("lokasi_teks wajib diisi")
    if not data.get("waktu_kejadian"):
        raise ValueError("waktu_kejadian wajib diisi")
    lat, lon = data.get("lat"), data.get("lon")
    if (lat is None) != (lon is None):
        raise ValueError("lat dan lon harus diisi berpasangan")
    # Data pribadi ditolak TEGAS, bukan diam-diam dibuang: pengirim harus tahu
    # bahwa kolom itu tidak disimpan, supaya tidak mengira datanya aman di sini.
    for pribadi in ("nama", "nama_pelapor", "telepon", "hp", "nik", "email", "alamat_pelapor"):
        if data.get(pribadi):
            raise ValueError(f"kolom '{pribadi}' tidak diterima: data pribadi pelapor tetap di "
                             "sistem asal (UU 27/2022); kirim nomor tiketnya di ref_sumber")
    return {
        "sumber": sumber,
        "ref_sumber": (str(data.get("ref_sumber")).strip() or None) if data.get("ref_sumber") else None,
        "kategori": kategori,
        "deskripsi": (str(data.get("deskripsi") or "").strip() or None),
        "lokasi_teks": lokasi,
        "wilayah": (str(data.get("wilayah") or "").strip() or None),
        "lat": float(lat) if lat is not None else None,
        "lon": float(lon) if lon is not None else None,
        "kamera_dugaan": (str(data.get("kamera_dugaan") or "").strip() or None),
        "waktu_kejadian": _iso(data["waktu_kejadian"]),
        "waktu_lapor": _iso(data.get("waktu_lapor") or data["waktu_kejadian"]),
    }


def buat(data: dict, oleh: str = "petugas") -> tuple[int, bool]:
    """Simpan satu aduan. Mengembalikan (id, baru). Aduan dengan pasangan
    (sumber, ref_sumber) yang sudah ada tidak digandakan."""
    d = validasi(data)
    kini = _kini()
    with store._lock, store._connect() as conn:
        if d["ref_sumber"]:
            ada = conn.execute("SELECT id FROM aduan WHERE sumber = ? AND ref_sumber = ?",
                               (d["sumber"], d["ref_sumber"])).fetchone()
            if ada:
                return int(ada["id"]), False
        cur = conn.execute(
            "INSERT INTO aduan (sumber, ref_sumber, kategori, deskripsi, lokasi_teks, wilayah,"
            " lat, lon, kamera_dugaan, waktu_kejadian, waktu_lapor, status, dibuat, diubah)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,'baru',?,?) RETURNING id",
            (d["sumber"], d["ref_sumber"], d["kategori"], d["deskripsi"], d["lokasi_teks"],
             d["wilayah"], d["lat"], d["lon"], d["kamera_dugaan"], d["waktu_kejadian"],
             d["waktu_lapor"], kini, kini))
        aid = int(cur.fetchone()[0])
        _riwayat(conn, aid, None, "baru", oleh, f"diterima dari {d['sumber']}")
    return aid, True


# -------------------------------------------------------------- pencocokan
def _kamera_calon(conn, a, kamera: list[dict]) -> list[tuple[str, str, float]]:
    """(key, cara, bobot) kamera yang mungkin meliput lokasi aduan."""
    if a["kamera_dugaan"]:
        return [(a["kamera_dugaan"], "ditunjuk petugas", BOBOT["kamera_ditunjuk"])]
    calon: list[tuple[str, str, float]] = []
    if a["lat"] is not None:
        for r in conn.execute("SELECT kamera, lat, lon, radius_m FROM kamera_lokasi"):
            d = _jarak_m(a["lat"], a["lon"], r["lat"], r["lon"])
            if d <= r["radius_m"]:
                calon.append((r["kamera"], f"koordinat, {d:.0f} m", BOBOT["kamera_koordinat"]))
    if calon:
        return calon
    kata = _kata(a["lokasi_teks"])
    for k in kamera:
        if a["wilayah"] and k.get("region") and a["wilayah"] != k["region"]:
            continue
        sama = kata & (_kata(k.get("name", "")) | _kata(k.get("area", "")))
        if sama:
            calon.append((k["key"], "nama lokasi: " + ", ".join(sorted(sama)), BOBOT["kamera_teks"]))
    return calon


def cocokkan(aduan_id: int, kamera: list[dict], jendela_menit: int = JENDELA_MENIT) -> dict:
    """Cari kejadian yang cocok untuk satu aduan dan perbarui statusnya."""
    with store._lock, store._connect() as conn:
        a = conn.execute("SELECT * FROM aduan WHERE id = ?", (aduan_id,)).fetchone()
        if not a:
            raise KeyError(aduan_id)
        if a["status"] in ("terbukti", "selesai"):
            return {"status": a["status"], "dilewati": "aduan sudah diputus"}

        t = datetime.fromisoformat(a["waktu_kejadian"])
        awal = (t - timedelta(minutes=jendela_menit)).isoformat(timespec="seconds")
        akhir = (t + timedelta(minutes=jendela_menit)).isoformat(timespec="seconds")
        jenis = KATEGORI_KE_JENIS.get(a["kategori"], ())
        calon = _kamera_calon(conn, a, kamera)
        keys = [c[0] for c in calon]

        siklus: dict[str, int] = {}
        if keys:
            tanda = ",".join("?" * len(keys))
            for r in conn.execute(
                    f"SELECT camera, COUNT(*) n FROM cycles WHERE ok = TRUE AND camera IN ({tanda})"
                    " AND finished_at >= ? AND finished_at <= ? GROUP BY camera",
                    (*keys, awal, akhir)):
                siklus[r["camera"]] = r["n"]

        kandidat = []
        if keys and jenis:
            tk, tj = ",".join("?" * len(keys)), ",".join("?" * len(jenis))
            baris = conn.execute(
                f"SELECT * FROM events WHERE camera IN ({tk}) AND kind IN ({tj})"
                " AND detected_at >= ? AND detected_at <= ?"
                " AND NOT (verified = TRUE AND verdict = 'bukan')",
                (*keys, *jenis, awal, akhir)).fetchall()
            bobot_kamera = {c[0]: (c[1], c[2]) for c in calon}
            for e in baris:
                dt = abs((datetime.fromisoformat(e["detected_at"]) - t).total_seconds()) / 60
                cara, bk = bobot_kamera[e["camera"]]
                rinci = {
                    "jenis_cocok": BOBOT["jenis"],
                    "kamera": bk, "cara_kamera": cara,
                    "waktu": round(BOBOT["waktu"] * max(0.0, 1 - dt / jendela_menit), 3),
                    "selisih_menit": round(dt, 1),
                    "terverifikasi_benar": BOBOT["terverifikasi"]
                    if (e["verified"] and e["verdict"] == "benar") else 0.0,
                }
                skor = min(1.0, rinci["jenis_cocok"] + rinci["kamera"] + rinci["waktu"]
                           + rinci["terverifikasi_benar"])
                kandidat.append((e["id"], round(skor, 3), rinci))

        # Kandidat lama yang belum diputus diganti hasil terbaru; yang sudah
        # diputus petugas dipertahankan apa adanya.
        conn.execute("DELETE FROM aduan_kecocokan WHERE aduan_id = ? AND keputusan = 'kandidat'",
                     (aduan_id,))
        for eid, skor, rinci in kandidat:
            conn.execute(
                "INSERT INTO aduan_kecocokan (aduan_id, event_id, skor, alasan, dibuat)"
                " VALUES (?,?,?,?,?) ON CONFLICT (aduan_id, event_id) DO NOTHING", (aduan_id, eid, skor, json.dumps(rinci), _kini()))

        terekam = sum(siklus.values())
        if not keys:
            baru, sebab = "di_luar_cakupan", "tidak ada kamera yang meliput lokasi ini"
        elif kandidat:
            baru, sebab = "ada_kandidat", f"{len(kandidat)} kejadian cocok, menunggu petugas"
        elif not jenis:
            baru, sebab = "tidak_terpantau", "kategori ini belum dideteksi otomatis oleh sistem"
        elif terekam:
            baru, sebab = "tidak_ditemukan", f"kamera terekam {terekam} kali pada rentang ini, tanpa kejadian cocok"
        else:
            baru, sebab = "tidak_terpantau", "tidak ada rekaman pada rentang waktu ini"

        cakupan = {"kamera": [{"key": c[0], "cara": c[1], "siklus_terekam": siklus.get(c[0], 0)}
                              for c in calon],
                   "rentang_utc": [awal, akhir], "jendela_menit": jendela_menit,
                   "jenis_dicari": list(jenis), "keterangan": sebab}
        conn.execute("UPDATE aduan SET status = ?, cakupan = ?, diubah = ? WHERE id = ?",
                     (baru, json.dumps(cakupan), _kini(), aduan_id))
        if baru != a["status"]:
            _riwayat(conn, aduan_id, a["status"], baru, "sistem", sebab)
    return {"status": baru, "keterangan": sebab, "kandidat": len(kandidat), "cakupan": cakupan}


def putus(aduan_id: int, event_id: int, keputusan: str, catatan: str | None = None) -> dict:
    """Keputusan petugas atas satu kandidat: 'dikonfirmasi' atau 'ditolak'."""
    if keputusan not in ("dikonfirmasi", "ditolak"):
        raise ValueError("keputusan harus 'dikonfirmasi' atau 'ditolak'")
    with store._lock, store._connect() as conn:
        a = conn.execute("SELECT * FROM aduan WHERE id = ?", (aduan_id,)).fetchone()
        if not a:
            raise KeyError(aduan_id)
        cur = conn.execute(
            "UPDATE aduan_kecocokan SET keputusan = ?, diputus_pada = ?, catatan = ?"
            " WHERE aduan_id = ? AND event_id = ?",
            (keputusan, _kini(), catatan, aduan_id, event_id))
        if not cur.rowcount:
            raise KeyError(f"kejadian {event_id} bukan kandidat aduan {aduan_id}")
        if keputusan == "dikonfirmasi":
            # Paling banyak satu bukti per aduan.
            conn.execute("UPDATE aduan_kecocokan SET keputusan = 'kandidat', diputus_pada = NULL"
                         " WHERE aduan_id = ? AND event_id != ? AND keputusan = 'dikonfirmasi'",
                         (aduan_id, event_id))
            baru, sebab = "terbukti", f"dibuktikan oleh kejadian #{event_id}"
        else:
            sisa = conn.execute("SELECT COUNT(*) FROM aduan_kecocokan WHERE aduan_id = ?"
                                " AND keputusan != 'ditolak'", (aduan_id,)).fetchone()[0]
            if sisa:
                baru, sebab = a["status"], None
            else:
                cak = json.loads(a["cakupan"] or "{}")
                terekam = sum(k.get("siklus_terekam", 0) for k in cak.get("kamera", []))
                baru = "tidak_ditemukan" if terekam else "tidak_terpantau"
                sebab = "semua kandidat ditolak petugas"
        conn.execute("UPDATE aduan SET status = ?, diubah = ? WHERE id = ?", (baru, _kini(), aduan_id))
        if baru != a["status"]:
            _riwayat(conn, aduan_id, a["status"], baru, "petugas", catatan or sebab)
    return {"id": aduan_id, "status": baru}


def ubah_status(aduan_id: int, status: str, catatan: str | None = None) -> dict:
    if status not in STATUS:
        raise ValueError(f"status tidak dikenal: {status!r}")
    with store._lock, store._connect() as conn:
        a = conn.execute("SELECT status FROM aduan WHERE id = ?", (aduan_id,)).fetchone()
        if not a:
            raise KeyError(aduan_id)
        conn.execute("UPDATE aduan SET status = ?, catatan = COALESCE(?, catatan), diubah = ?"
                     " WHERE id = ?", (status, catatan, _kini(), aduan_id))
        if status != a["status"]:
            _riwayat(conn, aduan_id, a["status"], status, "petugas", catatan)
    return {"id": aduan_id, "status": status}


# ------------------------------------------------------------------- baca
def daftar(status: str | None = None, page: int = 1, per_page: int = 20) -> dict:
    page, per_page = max(1, int(page)), min(100, max(5, int(per_page)))
    where, params = ("WHERE status = ?", [status]) if status else ("", [])
    with store._lock, store._connect() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM aduan {where}", params).fetchone()[0]
        rows = conn.execute(
            f"SELECT a.*, (SELECT COUNT(*) FROM aduan_kecocokan k WHERE k.aduan_id = a.id"
            f" AND k.keputusan != 'ditolak') AS n_kandidat FROM aduan a {where}"
            " ORDER BY waktu_kejadian DESC, id DESC LIMIT ? OFFSET ?",
            params + [per_page, (page - 1) * per_page]).fetchall()
        jumlah = {r["status"]: r["n"] for r in conn.execute(
            "SELECT status, COUNT(*) n FROM aduan GROUP BY status")}
    return {"items": [_baris(r) for r in rows], "total": total, "page": page,
            "per_page": per_page, "pages": max(1, -(-total // per_page)), "jumlah": jumlah}


def _baris(r) -> dict:
    d = dict(r)
    d["label_kategori"] = LABEL_KATEGORI.get(d["kategori"], d["kategori"])
    if d.get("cakupan"):
        d["cakupan"] = json.loads(d["cakupan"])
    return d


def detail(aduan_id: int) -> dict:
    with store._lock, store._connect() as conn:
        a = conn.execute("SELECT * FROM aduan WHERE id = ?", (aduan_id,)).fetchone()
        if not a:
            raise KeyError(aduan_id)
        kand = conn.execute(
            "SELECT k.*, e.camera, e.kind, e.track_id, e.detected_at, e.duration_sec,"
            " e.verified, e.verdict, e.authority FROM aduan_kecocokan k"
            " JOIN events e ON e.id = k.event_id WHERE k.aduan_id = ?"
            " ORDER BY (k.keputusan = 'dikonfirmasi') DESC, k.skor DESC", (aduan_id,)).fetchall()
        riw = conn.execute("SELECT * FROM aduan_riwayat WHERE aduan_id = ? ORDER BY id",
                           (aduan_id,)).fetchall()
    out = _baris(a)
    out["kandidat"] = [{**dict(k), "alasan": json.loads(k["alasan"]),
                        "label": store.LABEL.get(k["kind"], k["kind"])} for k in kand]
    out["riwayat"] = [dict(r) for r in riw]
    return out


# ------------------------------------------------------------ per kamera
def statistik_kamera(kamera: list[dict]) -> list[dict]:
    """Ukuran per kamera: aduan, ketepatan deteksi, cakupan rekaman.

    Ketepatan (presisi) = kejadian yang diputus 'benar' dibagi seluruh kejadian
    yang sudah diputus. Kejadian yang belum diputus tidak ikut, supaya angka
    ini tidak dipoles oleh antrean yang belum disentuh.
    """
    kini = datetime.now(timezone.utc)
    h1 = (kini - timedelta(days=1)).isoformat(timespec="seconds")
    h7 = (kini - timedelta(days=7)).isoformat(timespec="seconds")
    with store._lock, store._connect() as conn:
        ev = {r["camera"]: dict(r) for r in conn.execute(
            "SELECT camera, COUNT(*) total,"
            " SUM(CASE WHEN verified = TRUE AND verdict = 'benar' THEN 1 ELSE 0 END) benar,"
            " SUM(CASE WHEN verified = TRUE AND verdict = 'bukan' THEN 1 ELSE 0 END) bukan,"
            " SUM(CASE WHEN verified = FALSE THEN 1 ELSE 0 END) belum FROM events GROUP BY camera")}
        cy = {r["camera"]: dict(r) for r in conn.execute(
            "SELECT camera, SUM(CASE WHEN finished_at >= ? THEN 1 ELSE 0 END) h1,"
            " SUM(CASE WHEN finished_at >= ? THEN 1 ELSE 0 END) h7,"
            " SUM(CASE WHEN ok = FALSE AND finished_at >= ? THEN 1 ELSE 0 END) gagal7,"
            " MAX(finished_at) terakhir FROM cycles GROUP BY camera", (h1, h7, h7))}
        # Aduan dikaitkan ke kamera lewat kamera yang ditunjuk ATAU bukti yang
        # dikonfirmasi - bukan lewat kandidat mentah, yang bisa banyak kamera.
        ad: dict[str, dict] = {}
        for r in conn.execute(
                "SELECT a.status, COALESCE(e.camera, a.kamera_dugaan) cam FROM aduan a"
                " LEFT JOIN aduan_kecocokan k ON k.aduan_id = a.id AND k.keputusan = 'dikonfirmasi'"
                " LEFT JOIN events e ON e.id = k.event_id"):
            if not r["cam"]:
                continue
            s = ad.setdefault(r["cam"], {"total": 0})
            s["total"] += 1
            s[r["status"]] = s.get(r["status"], 0) + 1
    out = []
    for k in kamera:
        e = ev.get(k["key"], {}); c = cy.get(k["key"], {}); a = ad.get(k["key"], {"total": 0})
        diputus = (e.get("benar") or 0) + (e.get("bukan") or 0)
        out.append({
            "key": k["key"], "name": k.get("name"),
            "kejadian": {"total": e.get("total") or 0, "benar": e.get("benar") or 0,
                         "bukan": e.get("bukan") or 0, "belum": e.get("belum") or 0,
                         "presisi": round((e.get("benar") or 0) / diputus, 3) if diputus else None},
            "aduan": a,
            "rekaman": {"siklus_24j": c.get("h1") or 0, "siklus_7h": c.get("h7") or 0,
                        "gagal_7h": c.get("gagal7") or 0, "terakhir": c.get("terakhir")},
        })
    return out


def lokasi_simpan(kamera: str, lat: float, lon: float, radius_m: int = 150,
                  alamat: str | None = None) -> dict:
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("koordinat tidak sah")
    if not (10 <= int(radius_m) <= 2000):
        raise ValueError("radius_m harus 10-2000")
    with store._lock, store._connect() as conn:
        conn.execute(
            "INSERT INTO kamera_lokasi (kamera, lat, lon, radius_m, alamat, diubah) VALUES (?,?,?,?,?,?)"
            " ON CONFLICT(kamera) DO UPDATE SET lat = excluded.lat, lon = excluded.lon,"
            " radius_m = excluded.radius_m, alamat = excluded.alamat, diubah = excluded.diubah",
            (kamera, lat, lon, int(radius_m), alamat, _kini()))
    return {"kamera": kamera, "lat": lat, "lon": lon, "radius_m": int(radius_m)}


def lokasi_semua() -> dict:
    with store._lock, store._connect() as conn:
        return {r["kamera"]: dict(r) for r in conn.execute("SELECT * FROM kamera_lokasi")}
