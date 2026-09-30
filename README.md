# PP-Vehicle

Wrapper di atas [PaddleDetection PP-Vehicle](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/deploy/pipeline/README_en.md)
(release/2.9): config profile/skenario sendiri, CLI, REST API, dan UI web kecil
untuk menjalankan analitik kendaraan dari video, gambar, kamera, atau RTSP.

Upstream tidak disalin — di-clone ke `vendor/PaddleDetection` oleh script setup,
dan kita memanggil `deploy/pipeline/pipeline.py` sebagai subprocess.

---

## Apa yang bisa dilakukan

| Skenario | Fitur | Modul yang aktif |
|---|---|---|
| `detection` | Deteksi kendaraan (bbox) | DET |
| `tracking` | Tracking multi-objek, ID unik per kendaraan | MOT |
| `attribute` | 10 warna + 9 tipe bodi | MOT + VEHICLE_ATTR |
| `plate` | Pengenalan plat nomor (PP-OCRv3) | MOT + VEHICLE_PLATE |
| `counting` | Hitung masuk/keluar lewat garis virtual | MOT |
| `break_in` | Hitung kendaraan masuk zona poligon | MOT |
| `illegal_parking` | Parkir liar di zona + baca plat | MOT + VEHICLE_PLATE |
| `retrograde` | Kendaraan lawan arah | MOT + VEHICLE_RETROGRADE |
| `press_line` | Pelanggaran marka lajur | MOT + VEHICLE_PRESSING |
| `violation_all` | Marka + lawan arah sekaligus | MOT + keduanya |
| `full` | Semua di atas dalam satu run | semua |

Skenario ada di [`configs/scenarios/`](configs/scenarios) — satu file YAML per
skenario, gampang ditambah.

### Input yang didukung

Gambar, folder gambar, video, folder video, kamera lokal, dan RTSP. Hasil bisa
di-*push* balik ke RTSP server (`--pushurl`).

---

## Setup

**Docker adalah jalur yang didukung**, untuk laptop maupun server.

```bash
docker compose -f docker/docker-compose.yml up --build
```

UI di `http://localhost:8000`, dashboard presentasi di `/demo`.
Bobot model diunduh sekali ke volume `models` dan bertahan lintas rebuild.

Untuk GPU server (butuh NVIDIA Container Toolkit):

```bash
docker compose -f docker/docker-compose.yml --profile gpu up --build ppvehicle-gpu
```

### Kenapa Docker, bukan venv lokal

Kode tracking PaddleDetection bergantung pada `lap`, `cython_bbox`, dan
`pycocotools` — tiga C-extension yang **tidak punya wheel untuk Python 3.12+**
(`lap` bahkan mem-pin `Requires-Python <3.12`). Di Windows dengan Python 3.13,
`pip install -r requirements.txt` tidak akan resolve sama sekali. Container
memakai Python 3.10 di Linux, tempat ketiganya tersedia.

### Install native (opsional)

Hanya untuk Linux/macOS dengan Python 3.9–3.11:

```bash
bash scripts/setup.sh               # CPU
bash scripts/setup.sh --gpu cu126   # GPU
```

Di Windows `scripts/setup.ps1` melakukan hal yang sama, dan akan berhenti
dengan pesan jelas kalau Python-nya 3.12+. Kalau punya Python 3.10 terpisah:

```bash
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -PythonExe C:\Python310\python.exe
```

---

## Pemakaian

Di dalam container, awali dengan `docker compose -f docker/docker-compose.yml exec ppvehicle`
(atau `run --rm ppvehicle` kalau service-nya belum jalan):

```bash
# apa saja yang tersedia
python -m lalin scenarios
python -m lalin models
python -m lalin doctor

# lihat config persis yang akan dipakai, tanpa menjalankan apa pun
python -m lalin preview --scenario retrograde --profile gpu

# jalankan
python -m lalin run --scenario tracking --source data/samples/jalan.mp4
python -m lalin run --scenario plate --source data/samples/jalan.mp4 --profile gpu
python -m lalin run --scenario counting --source rtsp://kamera/stream1
python -m lalin run --scenario detection --source data/samples/foto.jpg

# parkir liar di zona sendiri, ambang 10 detik
python -m lalin run --scenario illegal_parking --source data/samples/jalan.mp4 --region-polygon 600 300 1300 300 1300 800 600 800 --illegal-parking-time 10

# lawan arah dengan fence line manual
python -m lalin run --scenario retrograde --source data/samples/jalan.mp4 --fence-line 570 163 1030 752

# lihat perintah yang akan dijalankan tanpa menjalankannya
python -m lalin run --scenario full --source data/samples/jalan.mp4 --dry-run

# API + UI di http://127.0.0.1:8000
python -m lalin serve
```

Hasil setiap run masuk ke `runs/<timestamp>-<skenario>/`: `infer_cfg.yml` yang
dipakai, `lane_seg_config.yml`, `run.log`, dan `output/` berisi video/gambar
teranotasi.

### Dashboard presentasi

`/demo` menyajikan **Ruang Kendali Lalin** — konsol operator untuk demo ke
instansi: rail kamera, live view dengan overlay zona/objek, feed kejadian
dengan label kewenangan (ranah Dishub vs rujuk Ditlantas), dan kartu bukti
berisi dasar hukum serta tombol tindakan.

Sumbernya `src/lalin/web/dashboard.html`, ditulis sebagai fragmen tanpa
`<html>`/`<head>` supaya file yang sama bisa dipublikasikan sebagai halaman
mandiri; `/demo` membungkusnya dengan skeleton dokumen.

Cuplikan cadangan di `data/demo/` diambil dari portal CCTV publik DKI, satu
per kamera terpilih.

Tiga mode sumber: **Live** (frame kamera disegarkan tiap 3 dtk), **Analisis**
(frame ber-kotak dari pipeline — keluaran model sungguhan), dan **Cuplikan**
(snapshot tersimpan, untuk demo tanpa jaringan).

Chip di header bersifat **sadar-mode**: ia menyebutkan bagian mana saja yang
masih contoh dan menghapusnya satu per satu saat data nyata masuk. Tiap bagian
yang masih contoh juga membawa penanda kuning **CONTOH** di tempatnya, supaya
terlihat dari kursi belakang ruang rapat, bukan hanya oleh yang membaca header.

### Analisis batch berkala

Inferensi kontinu per kamera butuh GPU. Di CPU pipeline berjalan ±4 fps
(Xeon 16 core) atau ±1,8 fps (laptop i5), jadi
worker batch memperlakukan tiap kamera sebagai klip pendek yang direkam
terjadwal, dianalisis, lalu diterbitkan sebagai "tampilan terakhir kamera ini"
— bukan berpura-pura live. Dashboard menampilkan waktu analisisnya supaya
jedanya terlihat, bukan disembunyikan.

Kamera dan zonanya di [`configs/cameras.yml`](configs/cameras.yml).

```bash
python -m lalin batch --list              # daftar kamera
python -m lalin batch --camera jatibaru   # satu kamera
python -m lalin batch                     # satu siklus penuh
```

| Endpoint | Fungsi |
|---|---|
| `GET /api/batch` | status worker + hasil terakhir tiap kamera |
| `POST /api/batch/run` | picu satu siklus (atau `?key=<kamera>`) |
| `POST /api/batch/start` · `/stop` | jadwal berkala (`?interval_sec=`) |
| `GET /api/batch/frame/{key}` | frame teranotasi terakhir — keluaran model |

Tiap siklus: ffmpeg merekam `clip_seconds` dari HLS pada `analysis_fps`,
skenario dijalankan, frame teranotasi terakhir diekstrak, dan `results.json`
plus frame itu disimpan di `runs/_latest/`.

**`clip_seconds` harus minimal 3x `illegal_parking_time`.** Alarm baru bisa
menyala setelah ambang terlampaui, jadi klip 12 detik dengan ambang 8 detik
hanya menyisakan jendela 4 detik dan hampir selalu menghasilkan nol kejadian.

Terukur pada klip dan kamera yang sama:

| | laptop i5-8250U | server Xeon 16 core |
|---|---|---|
| 150 frame (Jati Baru) | 85 dtk | **37,5 dtk** |
| siklus penuh | 228 dtk | **114 dtk** |

Di server, penangkapan klip justru jadi leher botolnya: klip 20 detik butuh
20 detik waktu nyata, sementara analisisnya ~38 detik.

### Rekap & verifikasi

Tiap siklus batch dicatat ke SQLite di `runs/lalin.db` (stdlib, tanpa dependensi
baru). Dipakai SQLite dan bukan log append-only karena putusan verifikasi
petugas adalah *update* atas kejadian yang sudah ada, bukan fakta baru.

| Endpoint | Fungsi |
|---|---|
| `GET /api/stats` | angka rekap hari berjalan (KPI + grafik per jam) |
| `GET /api/events` | kejadian terbaru |
| `POST /api/events/{id}/verify?verdict=benar\|bukan` | putusan petugas |

Tidak ada deduplikasi antar siklus: tiap siklus menganalisis klip yang baru
direkam, jadi track id 2 di satu siklus dan track id 2 di siklus berikutnya
adalah kendaraan berbeda.

"Hari ini" dihitung dalam **WIB**, bukan UTC — rekap untuk instansi Jakarta yang
berganti hari pukul 07.00 pagi jelas salah. Jakarta UTC+7 sepanjang tahun tanpa
DST, jadi offset tetap dipakai agar tidak bergantung pada basis data zona waktu
di dalam container.

Dashboard memakai angka ini begitu ada minimal satu siklus hari itu; sebelum itu
kartu KPI tetap memakai nilai contoh dengan penanda kuning **CONTOH**. Chip di
header menyebutkan bagian mana saja yang masih contoh, dan menghapusnya satu per
satu saat data nyata masuk.

### Sumber CCTV publik DKI

Portal `jakcctv.jakarta.go.id` menyajikan 50 kamera lewat Flussonic. Stream-nya
HLS biasa, jadi bisa langsung dipakai sebagai `--source` tanpa RTSP:

```
https://dki-jkt.balitower.co.id:7028/<ID_KAMERA>/index.m3u8
https://dki-jkt.balitower.co.id:7028/<ID_KAMERA>/preview.jpg        # snapshot
https://dki-jkt.balitower.co.id:7028/<ID_KAMERA>/media_info.json    # spesifikasi
```

Spesifikasinya **640x360, 20 fps, H.264 534 kbps**. Cukup untuk deteksi,
tracking, counting, dan aturan berbasis zona; **tidak cukup untuk baca pelat**
— pada resolusi itu pelat hanya selebar 8-15 piksel, di bawah ambang OCR
(sekitar 100 piksel). Fitur pelat memerlukan akses feed sumber >=1080p.

### Editor zona visual — `/zona`

Klik titik di atas frame kamera untuk membentuk poligon, seret titik untuk
menggesernya, simpan. Berlaku untuk siklus batch berikutnya.

| Endpoint | Fungsi |
|---|---|
| `GET /api/cameras` | kamera + zona yang berlaku, termasuk asal-usulnya |
| `PUT /api/cameras/{key}/zone` | simpan poligon dari editor |
| `DELETE /api/cameras/{key}/zone` | buang timpaan, kembali ke `cameras.yml` |

Hasil editor **tidak ditulis balik ke `cameras.yml`** melainkan ke
`configs/zones.yml` yang menimpanya saat dimuat. Alasannya: `cameras.yml`
berisi 75 baris komentar — hasil survei kamera, kalibrasi fps, alasan tiap
ambang parkir berbeda — dan `yaml.safe_dump` akan menghapus semuanya.
Pemisahan ini juga membuat asal-usul tiap nilai jelas: yang ditulis tangan
tetap di `cameras.yml`, yang disetel operator ada di `zones.yml`. Badge di
editor menampilkan nilai mana yang sedang berlaku.

Titik di luar frame 640×360 ditolak API — zona di luar frame tidak akan
pernah memicu apa pun, dan gagal diam-diam jauh lebih merepotkan daripada
ditolak di muka.

### Memilih kamera (survei berbasis ukuran)

Memilih kamera dengan mata tidak berskala ke 46 feed, dan yang menentukan
kamera layak pakai bukan pemandangannya melainkan berapa piksel yang ditempati
kendaraan di dalamnya.

```bash
python scripts/survey_cameras.py --list kamera.txt
```

Menjalankan detector pada satu snapshot tiap kamera lalu memberi skor
`jumlah kendaraan x luas bbox median (% frame)`. Perkalian itu menghukum
"banyak kotak mungil" dan "satu kotak besar" sama beratnya. Hasil survei
22 Sep 2026 atas 46 kamera publik ada di `runs/_survey/survey.json`; kamera
terpilih beserta skornya tercatat di `configs/cameras.yml`.

### Bikin koordinat zona

Untuk lingkungan tanpa layar (Docker, SSH), `pick_zone.py` tidak bisa membuka
jendela. Pakai pendampingnya:

```bash
python scripts/grid_frame.py data/samples/klip.mp4 --frame 30 --step 40
```

Ia menempelkan grid koordinat di atas frame, sehingga poligon bisa dibaca
langsung dari gambarnya.

### Bikin koordinat zona (dengan GUI)

```bash
python scripts/pick_zone.py data/samples/jalan.mp4 --save zona_bahu_jalan
```

Klik titik searah jarum jam, `s` simpan. Lihat [`configs/zones/`](configs/zones).

---

## Profil perangkat

`configs/profiles/` memisahkan *di mana* dijalankan dari *apa* yang dihitung.

| | `cpu.yml` | `gpu.yml` |
|---|---|---|
| detector | PP-YOLOE-**s** (27 MB) | PP-YOLOE-**l** (182 MB) |
| run_mode | `paddle` + MKLDNN, 4 thread | `trt_fp16` |
| skip_frame_num | 2 | -1 (tanpa skip) |

Ganti profil per-run dengan `--profile gpu`, atau ubah default lewat
`PPVEHICLE_PROFILE` di `.env`.

---

## Arsitektur

```
configs/profiles/*.yml  ─┐
                         ├─► src/ppvehicle/config.py ─► runs/<id>/infer_cfg.yml
configs/scenarios/*.yml ─┘                            + argv flags
                                                          │
                                                          ▼
                                   subprocess: vendor/PaddleDetection/
                                               deploy/pipeline/pipeline.py
```

- `config.py` — menggabungkan profil + skenario jadi skema `infer_cfg_ppvehicle.yml`
  asli plus daftar flag CLI. Ini satu-satunya tempat yang tahu format upstream.
- `models.py` — registry model zoo + downloader ke `models/`. Kalau bobot belum
  ada, config jatuh balik ke URL bcebos dan PaddleDetection yang mengunduh.
- `runner.py` — spawn subprocess dengan `cwd` = root PaddleDetection (wajib:
  sebagian kode upstream me-resolve path `deploy/...` relatif ke cwd).
- `jobs.py` — registry job in-memory + semaphore concurrency.
- `api.py` / `web/index.html` — REST + UI.

### Kenapa subprocess, bukan import

`pipeline.py` melakukan manipulasi `sys.path` sendiri dan memanggil `sys.exit`.
Subprocess bikin API tetap hidup kalau pipeline mati, dan lognya bisa di-stream.

### Kenapa `lane_seg_config.yml` di-generate ulang

File bawaan upstream menulis `device: gpu` secara hardcoded. Di profil CPU itu
akan gagal, jadi `config.py` selalu menulis ulang file itu per-run mengikuti
profil yang dipakai.

---

## Batasan yang perlu diketahui

**Plat nomor dilatih untuk plat Cina.** Bobot `ch_PP-OCRv3_rec` berasal dari
dataset CCPD (format `京A·12345`, termasuk plat hijau EV). Untuk plat Indonesia
(`B 1234 XYZ`) akurasinya akan rendah. Dua jalur perbaikan:

1. **Cepat** — ganti model recognition ke PP-OCR latin/English, sesuaikan
   `word_dict_path` ke charset A–Z + 0–9. Bagus untuk uji kelayakan, belum tentu
   cukup untuk produksi.
2. **Benar** — fine-tune PP-OCRv3 rec dengan dataset plat Indonesia
   (beberapa ribu crop plat berlabel sudah cukup jauh). Ini pekerjaan tersendiri
   dengan anggaran waktu sendiri.

**Atribut kendaraan dari dataset VeRi** (Cina). Warna dan tipe umum masih
transferable, tapi tidak ada kelas untuk angkot, bajaj, atau bentor.

**`keep_right_flag` default kami `false`.** Upstream menganggap lalu lintas
jalur kanan; Indonesia jalur kiri. Sudah disetel di skenario `retrograde`,
`violation_all`, dan `full`.

**Kamera harus statis** untuk counting, break-in, parkir liar, dan lawan arah.

**Performa.** Angka 13–25 ms di tabel upstream diukur di Tesla T4 dengan
TensorRT. Di laptop CPU tanpa GPU (mis. i5-8250U) pipeline penuh realistis di
kisaran 0.5–2 FPS — cukup untuk uji kebenaran dan demo video offline, tidak
untuk RTSP realtime.

**`ffmpeg`** tidak wajib untuk inferensi video biasa, tapi dibutuhkan untuk
`--pushurl` dan sebagian encoding. Belum ada di PATH? Install lewat
`winget install Gyan.FFmpeg`.

**TensorRT di Docker.** `docker/Dockerfile.gpu` berbasis `nvidia/cuda` dan jalan
dengan `--run_mode paddle` di GPU. Untuk `trt_fp16` ganti baris `FROM` ke image
resmi PaddlePaddle yang sudah membundel TensorRT — lihat komentar di file itu.

---

## Model zoo

Semua diunduh dari `bj.bcebos.com/v1/paddledet/models/pipeline/`:

| Key | Tugas | Ukuran |
|---|---|---|
| `ppyoloe_l` | deteksi + MOT presisi tinggi | 182 MB |
| `ppyoloe_s` | deteksi + MOT ringan | 27 MB |
| `vehicle_attr` | atribut (PP-LCNet) | 6 MB |
| `plate_det` | deteksi teks plat (PP-OCRv3) | 2 MB |
| `plate_rec` | pengenalan teks plat (PP-OCRv3) | 9 MB |
| `lane_seg` | segmentasi lajur (PP-LiteSeg, BDD100K) | 44 MB |

```bash
python -m lalin models --download                  # semua
python -m lalin models --download ppyoloe_s plate_det plate_rec
```

---

## REST API

| Endpoint | Fungsi |
|---|---|
| `GET /api/doctor` | status environment, paddle, model |
| `GET /api/scenarios` | daftar skenario |
| `GET /api/profiles` | daftar profil perangkat |
| `GET /api/preview?scenario=…` | config + argv yang akan dipakai |
| `GET /api/models` · `POST /api/models/download` | kelola bobot |
| `POST /api/jobs` | mulai run (upload file atau `source`) |
| `GET /api/jobs` · `GET /api/jobs/{id}` | status + tail log |
| `GET /api/jobs/{id}/artifacts` · `?name=` | ambil hasil |

Dokumentasi interaktif di `/docs`.

---

## Lisensi

Kode di repo ini mengikuti lisensi projectmu. PaddleDetection dan bobot
modelnya berlisensi Apache 2.0 — lihat upstream.
