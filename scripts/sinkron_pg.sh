#!/bin/sh
# Salin isi SQLite (runs/lalin.db) ke PostgreSQL (kontainer docker-db-1).
#
# SEMENTARA: dipakai selama aplikasi masih menulis ke SQLite (tahap 1 migrasi).
# Setiap kali dijalankan, tabel PostgreSQL DIKOSONGKAN lalu dimuat ulang dari
# SQLite - jadi jangan jalankan lagi setelah aplikasi menulis langsung ke
# PostgreSQL (tahap 2), atau data yang hanya ada di PostgreSQL akan hilang.
#
# Jalankan di server:  sh /home/user3/ppvehicle/scripts/sinkron_pg.sh
set -eu

# Kunci pengaman: begitu aplikasi menulis langsung ke PostgreSQL, skrip ini
# akan MENGHAPUS data yang hanya ada di sana. Berkas penanda dibuat saat
# peralihan (tahap 2).
if [ -f /home/user3/ppvehicle/docker/.pg-utama ]; then
  echo "DITOLAK: PostgreSQL sudah menjadi basis data utama sejak $(cat /home/user3/ppvehicle/docker/.pg-utama)." >&2
  echo "Menjalankan sinkron akan menimpa data yang hanya ada di PostgreSQL." >&2
  exit 2
fi

APP=${APP:-docker-ppvehicle-1}   # bisa ditimpa: kontainer lain yang me-mount runs/
DB=docker-db-1
TMP=/tmp/lalin_sinkron
TABEL="cycles events aduan aduan_kecocokan aduan_riwayat kamera_lokasi"

rm -rf "$TMP" && mkdir -p "$TMP"

# 1. Ekspor per tabel ke CSV, kolom diambil dari SQLite apa adanya.
docker exec -i "$APP" python - <<'PY'
import csv, sqlite3, os
os.makedirs("/tmp/lalin_sinkron", exist_ok=True)
c = sqlite3.connect("/app/runs/lalin.db")
for t in ("cycles", "events", "aduan", "aduan_kecocokan", "aduan_riwayat", "kamera_lokasi"):
    cur = c.execute(f"SELECT * FROM {t}")
    kolom = [d[0] for d in cur.description]
    with open(f"/tmp/lalin_sinkron/{t}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(kolom)
        n = 0
        for r in cur:
            w.writerow(["" if v is None else v for v in r])
            n += 1
    with open(f"/tmp/lalin_sinkron/{t}.kolom", "w") as f:
        f.write(",".join(kolom))
    # Jumlah baris dicatat di sini, bukan dihitung dengan wc -l: kolom teks
    # (uraian aduan, JSON cakupan) bisa memuat baris baru.
    with open(f"/tmp/lalin_sinkron/{t}.n", "w") as f:
        f.write(str(n))
    print(f"ekspor {t:<16} {n:>6} baris")
PY
docker cp "$APP:/tmp/lalin_sinkron/." "$TMP/"
docker exec "$APP" rm -rf /tmp/lalin_sinkron
docker exec "$DB" rm -rf /tmp/lalin_sinkron
docker cp "$TMP" "$DB:/tmp/lalin_sinkron"

# 2. Kosongkan lalu muat, dalam SATU transaksi: gagal di tengah = tidak ada
#    yang berubah. Urutan mengikuti kunci asing.
{
  echo "BEGIN;"
  echo "TRUNCATE lalin.aduan_riwayat, lalin.aduan_kecocokan, lalin.aduan, lalin.events, lalin.cycles, lalin.kamera_lokasi RESTART IDENTITY;"
  for t in $TABEL; do
    k=$(cat "$TMP/$t.kolom")
    # printf, BUKAN echo: echo milik dash membaca "\c" pada "\copy" sebagai
    # perintah "hentikan keluaran", dan baris salinnya lenyap tanpa galat.
    printf '%s\n' "\\copy lalin.$t ($k) FROM '/tmp/lalin_sinkron/$t.csv' WITH (FORMAT csv, HEADER true)"
  done
  # Penghitung ID dilanjutkan dari ID terbesar, supaya baris baru tidak
  # bertabrakan dengan ID lama yang dirujuk foto bukti.
  for t in cycles events aduan aduan_kecocokan aduan_riwayat; do
    echo "SELECT setval(pg_get_serial_sequence('lalin.$t','id'), COALESCE((SELECT MAX(id) FROM lalin.$t), 0) + 1, false);"
  done
  echo "COMMIT;"
} > "$TMP/muat.sql"
docker cp "$TMP/muat.sql" "$DB:/tmp/lalin_sinkron/muat.sql"
docker exec "$DB" psql -v ON_ERROR_STOP=1 -q -U lalin_admin -d lalin -f /tmp/lalin_sinkron/muat.sql >/dev/null

# 3. Bandingkan jumlah baris kedua sisi.
echo "--- pemeriksaan jumlah baris (SQLite / PostgreSQL) ---"
for t in $TABEL; do
  a=$(cat "$TMP/$t.n")
  b=$(docker exec "$DB" psql -tA -U lalin_admin -d lalin -c "SELECT COUNT(*) FROM lalin.$t")
  s="OK"; [ "$a" = "$b" ] || s="BERBEDA"
  printf "%-16s %6s / %-6s %s\n" "$t" "$a" "$b" "$s"
done
docker exec "$DB" rm -rf /tmp/lalin_sinkron
rm -rf "$TMP"
