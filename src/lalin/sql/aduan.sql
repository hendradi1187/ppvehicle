-- =====================================================================
-- Aduan masyarakat dan pencocokannya dengan kejadian terekam
-- SQLite 3.35+ (dijalankan aplikasi saat start lewat store.init()).
-- Padanan PostgreSQL (seluruh skema, termasuk events/cycles): docs/ddl/skema_postgres.sql
--
-- Alur:
--   1. Aduan masuk dari CRM/JAKI (impor) atau diinput petugas     -> aduan
--   2. Sistem mencari kejadian pada kamera, rentang waktu, dan jenis
--      yang sesuai                                                 -> aduan_kecocokan
--   3. Sistem memeriksa apakah kamera itu MEMANG TEREKAM pada rentang
--      tersebut (tabel cycles) - ketiadaan kejadian tanpa rekaman
--      bukan bukti pelanggaran tidak terjadi
--   4. Petugas memutus: kandidat dikonfirmasi / ditolak            -> aduan.status
--   5. Setiap perubahan status dicatat                             -> aduan_riwayat
--
-- DATA PRIBADI: tidak ada kolom nama, telepon, NIK, atau alamat pelapor.
-- Semuanya tetap di sistem asal; di sini hanya nomor tiketnya (ref_sumber).
-- Prinsip minimisasi data UU No. 27/2022 tentang Pelindungan Data Pribadi.
--
-- Waktu: TEXT ISO-8601 UTC ('2026-09-25T12:41:31+00:00'), sama dengan tabel
-- events dan cycles. Tampilan mengonversi ke WIB.
-- =====================================================================

CREATE TABLE IF NOT EXISTS aduan (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Asal aduan dan nomor tiketnya di sistem asal. Pasangan ini unik, jadi
    -- impor ulang dari CRM tidak menggandakan aduan.
    sumber          TEXT    NOT NULL DEFAULT 'crm'
                    CHECK (sumber IN ('crm', 'jaki', 'telepon', 'medsos', 'petugas')),
    ref_sumber      TEXT,
    kategori        TEXT    NOT NULL
                    CHECK (kategori IN ('parkir_liar', 'lawan_arah', 'langgar_marka',
                                        'jalur_sepeda', 'ngetem', 'kemacetan', 'lainnya')),
    deskripsi       TEXT,
    -- Lokasi sebagaimana dilaporkan. Koordinat opsional: tidak semua aduan
    -- membawanya, dan koordinat dari ponsel pelapor bisa meleset puluhan meter.
    lokasi_teks     TEXT    NOT NULL,
    wilayah         TEXT,
    lat             REAL    CHECK (lat IS NULL OR lat BETWEEN -90 AND 90),
    lon             REAL    CHECK (lon IS NULL OR lon BETWEEN -180 AND 180),
    -- Kamera yang ditunjuk petugas saat menerima aduan (key di cameras.yml).
    kamera_dugaan   TEXT,
    waktu_kejadian  TEXT    NOT NULL,          -- waktu kejadian menurut pelapor
    waktu_lapor     TEXT    NOT NULL,          -- waktu aduan dibuat di sistem asal
    -- baru              : belum dicocokkan
    -- ada_kandidat      : ditemukan kejadian yang cocok, menunggu petugas
    -- terbukti          : petugas mengonfirmasi satu kejadian sebagai bukti
    -- tidak_ditemukan   : kamera TEREKAM pada rentangnya, tidak ada kejadian cocok
    -- tidak_terpantau   : tidak ada rekaman pada rentangnya - tidak bisa dinilai
    -- di_luar_cakupan   : lokasi tidak diliput kamera mana pun
    -- selesai           : ditutup petugas
    status          TEXT    NOT NULL DEFAULT 'baru'
                    CHECK (status IN ('baru', 'ada_kandidat', 'terbukti', 'tidak_ditemukan',
                                      'tidak_terpantau', 'di_luar_cakupan', 'selesai')),
    -- Ringkasan cakupan rekaman hasil pencocokan terakhir (JSON): kamera yang
    -- diperiksa, jumlah siklus rekaman di rentang waktu, rentang yang dipakai.
    cakupan         TEXT,
    catatan         TEXT,
    dibuat          TEXT    NOT NULL,
    diubah          TEXT    NOT NULL,
    UNIQUE (sumber, ref_sumber)
);
CREATE INDEX IF NOT EXISTS idx_aduan_waktu  ON aduan(waktu_kejadian);
CREATE INDEX IF NOT EXISTS idx_aduan_status ON aduan(status);

-- Kandidat kejadian untuk satu aduan. Satu aduan bisa punya banyak kandidat;
-- paling banyak satu yang 'dikonfirmasi' (dijaga di aplikasi).
CREATE TABLE IF NOT EXISTS aduan_kecocokan (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    aduan_id        INTEGER NOT NULL REFERENCES aduan(id) ON DELETE CASCADE,
    event_id        INTEGER NOT NULL REFERENCES events(id),
    -- 0..1. Komponen dan bobotnya tersimpan di `alasan` supaya skor bisa
    -- diaudit, bukan angka ajaib.
    skor            REAL    NOT NULL CHECK (skor BETWEEN 0 AND 1),
    alasan          TEXT    NOT NULL,          -- JSON
    keputusan       TEXT    NOT NULL DEFAULT 'kandidat'
                    CHECK (keputusan IN ('kandidat', 'dikonfirmasi', 'ditolak')),
    diputus_pada    TEXT,
    catatan         TEXT,
    dibuat          TEXT    NOT NULL,
    UNIQUE (aduan_id, event_id)
);
CREATE INDEX IF NOT EXISTS idx_kecocokan_event ON aduan_kecocokan(event_id);

-- Jejak audit perubahan status aduan. Tidak pernah diubah atau dihapus.
CREATE TABLE IF NOT EXISTS aduan_riwayat (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    aduan_id        INTEGER NOT NULL REFERENCES aduan(id) ON DELETE CASCADE,
    waktu           TEXT    NOT NULL,
    dari_status     TEXT,
    ke_status       TEXT    NOT NULL,
    oleh            TEXT    NOT NULL DEFAULT 'sistem',   -- 'sistem' | 'petugas'
    catatan         TEXT
);
CREATE INDEX IF NOT EXISTS idx_riwayat_aduan ON aduan_riwayat(aduan_id);

-- Koordinat liputan kamera, untuk mencocokkan aduan berkoordinat dengan kamera
-- terdekat. Diisi petugas dari lokasi pemasangan yang sebenarnya; sengaja
-- tidak diisi otomatis - koordinat tebakan dari nama jalan bisa menunjuk
-- ke simpang yang salah.
CREATE TABLE IF NOT EXISTS kamera_lokasi (
    kamera          TEXT    PRIMARY KEY,       -- key di cameras.yml
    lat             REAL    NOT NULL CHECK (lat BETWEEN -90 AND 90),
    lon             REAL    NOT NULL CHECK (lon BETWEEN -180 AND 180),
    radius_m        INTEGER NOT NULL DEFAULT 150 CHECK (radius_m BETWEEN 10 AND 2000),
    alamat          TEXT,
    diubah          TEXT    NOT NULL
);
