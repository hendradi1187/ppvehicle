# Blueprint Optimasi PPVehicle CPU — 2 Kamera PoC

**Lokasi target:** workspace PPVehicle CPU `/home/user3/ppvehicle`  
**Kamera:** `benhil2` dan `gerbangpemuda`  
**Batasan:** CPU-only. Dokumen ini tidak menjadikan GPU sebagai solusi atau pembanding.

## 1. Sasaran akhir yang wajib dicapai

Untuk dua kamera, sistem harus berjalan terus-menerus dan menghasilkan satu rantai data yang dapat diaudit:

1. mengambil video tanpa putus;
2. menemukan setiap objek lalu lintas yang terlihat, termasuk kendaraan berkecepatan tinggi;
3. menjaga ID objek tetap stabil selama berada di layar;
4. memberi kelas kendaraan yang sesuai taksonomi Dishub;
5. menghitung kendaraan tepat satu kali ketika melintasi garis;
6. menghitung kepadatan, arus, kecepatan relatif, antrean, dan status kemacetan;
7. mendeteksi pelanggaran yang memang telah dikonfigurasi per kamera;
8. menyimpan bbox, track, crossing, metrik, event, waktu, kamera, confidence, dan bukti;
9. tidak membuang objek diam-diam: objek yang belum dapat dipilah harus tetap dihitung sebagai `UNKNOWN` dan masuk total lalu lintas.

`100% semua kendaraan` tidak boleh hanya berarti semua kotak tampil di dashboard. Definisi lulusnya harus berdasarkan video referensi yang dilabel manual dan metrik pada Bagian 11.

## 2. Kondisi aktual hasil audit 29 September 2026

### Infrastruktur

- CPU: 16 vCPU Intel Xeon Gold 5218, AVX2 dan AVX-512 tersedia, 2 NUMA node.
- RAM: 19 GiB, sekitar 13 GiB masih tersedia saat audit, tanpa swap.
- Container `docker-ppvehicle-1`: batas 8 CPU dan 5 GiB RAM.
- Runtime: Python 3.10.21, Paddle 3.3.1 CPU, MKLDNN aktif.
- Model tersedia: PP-YOLOE-S/L PPVehicle, model atribut, pelat, dan lane segmentation.
- Service sedang idle hanya memakai sekitar 0,16% CPU; host sekitar 91–93% idle saat tidak inferensi.
- Storage `runs/` sudah sekitar 32 GiB. Ini perlu retensi, tetapi bukan penyebab kendaraan terlewat.

### Bukti bottleneck saat pipeline bekerja

| Tahap | Bukti aktual | Dampak |
|---|---|---|
| Capture | Run terakhir Benhil gagal `Connection timed out`/TLS EOF. Gerbang Pemuda juga timeout | Tidak ada frame, sehingga model sebaik apa pun menghasilkan nol |
| Detector/MOT | Sekitar 124–125 ms/frame | Kapasitas detector sekitar 8 FPS |
| Pipeline Benhil | 138 ms/frame, sekitar 7,2 FPS | Di bawah input 10 FPS |
| Pipeline lengkap Gerbang Pemuda | 181–224 ms/frame, sekitar 4,5–5,5 FPS | Atribut dan OCR membuat throughput hampir separuh input |
| Model/taksonomi | Model COCO tidak memiliki kelas `kendaraan_sedang`; bus dan truk masih ambigu | Tujuh kelas Dishub belum mungkin akurat dengan model sekarang |
| Counting | Benhil pernah menghasilkan 171 unique tracks tetapi hanya 0 crossing pada counter; pipeline lain menghasilkan 5 crossing | Garis, tracker, dan definisi crossing belum tervalidasi bersama |
| Kemacetan | `aduan.py` menyatakan kemacetan belum dideteksi otomatis | Engine kemacetan belum tersedia, bukan masalah performa CPU |
| Pelanggaran | Mode PoC aktif menjeda siklus pelanggaran; run contoh dua kamera menghasilkan 0 event | Fitur ada sebagian, tetapi belum berjalan dan belum tervalidasi end-to-end |

**Putusan audit:** hardware belum terbukti sebagai batas utama. Urutan hambatan saat ini adalah: **capture → arsitektur pipeline ganda → throughput model → taksonomi model → kalibrasi tracking/counting → engine kemacetan/pelanggaran → baru kapasitas hardware**.

## 3. Masalah arsitektur existing

Alur sekarang menjalankan beberapa pekerjaan yang tumpang tindih:

```text
HLS kamera
  └─ FFmpeg capture + resize + encode klip
       ├─ PPVehicle pipeline: detection + MOT + atribut/pelat/lane/pelanggaran
       └─ mesin hitung terpisah: detection + JDETracker + klasifikasi/counting
```

Akibatnya, frame yang sama dapat didekode dan dideteksi lebih dari sekali. Gerbang Pemuda juga selalu membayar biaya atribut/pelat walaupun modul tersebut tidak diperlukan untuk menghitung kendaraan atau menentukan macet. Ini pemborosan CPU yang lebih dulu harus dihilangkan sebelum menaikkan jatah core.

Pipeline berbasis klip juga membuat hasil terlambat. Jika pemrosesan 4,5 FPS menerima klip 10 FPS, data tidak langsung hilang karena tersimpan di klip, tetapi latensinya terus tertinggal. Pada mode real-time dengan antrean biasa, backlog akan terus membesar.

## 4. Flow CPU target

```text
benhil2 HLS --------┐
                    ├─ Capture supervisor
gerbangpemuda HLS --┘    ├─ timeout, reconnect, health, jitter metric
                         └─ timestamp sumber + sequence frame
                                  │
                                  ▼
                    Decode sekali per kamera (FFmpeg)
                    640x360/960x540, 10–12 FPS terukur
                                  │
                                  ▼
                    Bounded latest-frame buffer
                    maksimal 1–2 frame; frame tua dibuang
                                  │
                                  ▼
                    ROI jalan per kamera
                    area langit/trotoar yang tidak relevan tidak diinferensi
                                  │
                                  ▼
                    Satu detector CPU teroptimasi
                    model compact khusus kelas Dishub
                    oneDNN/MKLDNN + thread/NUMA terukur
                                  │
                                  ▼
                    Tracker per kamera
                    Kalman/ByteTrack + interpolasi antar-detection
                    ID stabil, short-gap recovery, class voting per track
                                  │
                 ┌────────────────┼──────────────────┐
                 ▼                ▼                  ▼
          Line crossing     Traffic state      Violation rules
          arah + debounce   flow/occupancy      ROI/direction/stop/
          satu ID satu hit  speed/queue         lane/parking rules
                 │                │                  │
                 └────────────────┼──────────────────┘
                                  ▼
                    Event aggregator + quality gate
                    dedup, confidence, evidence, revision
                                  │
                                  ▼
                    PostgreSQL + API :8100 + dashboard
```

Prinsipnya: **decode sekali, detect sekali, gunakan hasil track yang sama untuk counting, macet, dan pelanggaran**.

## 5. Prioritas implementasi

### P0 — Stabilkan capture terlebih dahulu

Capture adalah blocker pertama karena run terbaru kedua kamera gagal sebelum inferensi.

- Buat satu capture supervisor per kamera, bukan `subprocess.run()` sekali lalu gagal satu siklus.
- Tambahkan reconnect dengan exponential backoff pendek, health state, dan counter putus.
- Bedakan `source_down`, `timeout`, `decode_error`, dan `no_new_frame`.
- Simpan `source_timestamp`, `received_at`, `frame_age_ms`, sequence, dan gap frame.
- Gunakan buffer terbaru berukuran 1–2; jangan antrekan frame lama.
- Pastikan URL/substream stabil. Jika sumber hanya HLS, ukur umur segmen; jangan menyebut real-time bila frame sudah terlambat beberapa detik.
- Sediakan rekaman referensi 15–30 menit dari tiap kamera agar tuning tetap bisa dilakukan ketika sumber live mati.

Kriteria selesai: kedua feed bertahan minimal 2 jam, availability minimal 99%, reconnect otomatis, dan tidak ada gap yang tidak tercatat.

### P1 — Satukan pipeline

- Hapus deteksi ganda antara pipeline pelanggaran dan mesin hitung.
- Detector menghasilkan satu `TrackObservation` canonical yang dikonsumsi semua modul.
- Atribut dan OCR dibuat **on-demand**, hanya ketika event membutuhkan bukti tersebut. Jangan jalankan OCR untuk seluruh kendaraan setiap frame.
- Lane segmentation dijalankan saat startup/kalibrasi atau periodik, bukan wajib per objek bila kamera statis.
- Tulis ke DB secara batch/asinkron supaya inferensi tidak menunggu I/O.

Perkiraan ruang yang bisa direbut kembali terlihat dari Gerbang Pemuda: detector/MOT sekitar 131 ms/frame, sedangkan pipeline lengkap menjadi 181–224 ms/frame. Cabang atribut/pelat harus keluar dari hot path.

### P2 — Optimasi detector CPU secara terukur

Urutan eksperimen, satu perubahan per run:

1. baseline Paddle + MKLDNN saat ini;
2. ROI crop sebelum resize;
3. uji thread 4, 6, 8 dan pin satu NUMA node; jangan berasumsi thread terbanyak selalu tercepat;
4. matikan visualisasi/encoding pada hot path;
5. bandingkan Paddle, ONNX Runtime, dan OpenVINO pada model yang sama;
6. uji FP32 vs INT8 setelah kalibrasi akurasi;
7. bandingkan input 640x360 dan 960x540 untuk objek jauh/kecil;
8. uji dua kamera bersamaan dengan scheduler fair.

Environment yang perlu diuji, bukan langsung dipatok:

```text
OMP_NUM_THREADS=4|6|8
MKL_NUM_THREADS=4|6|8
KMP_AFFINITY=granularity=fine,compact,1,0
cpuset satu NUMA node: 0-7 atau 8-15
```

Xeon ini mendukung AVX-512, tetapi container saat ini bebas berpindah di dua NUMA node. Pinning dapat mengurangi perpindahan cache/memori. Hasil harus dibuktikan lewat p50/p95, bukan hanya rata-rata FPS.

### P3 — Model khusus kelas Dishub

Model COCO saat ini hanya memberi label dasar `motorcycle`, `car`, `bus`, `truck`, `person`, dan `bicycle`. Ia tidak dapat membedakan:

- kendaraan sedang vs bus besar;
- pikap/truk kecil vs truk 2 sumbu vs truk berat;
- jenis lokal seperti angkot/Mikrotrans bila tidak dilatih.

Solusi CPU bukan memakai model makin besar secara buta. Gunakan detector compact yang di-fine-tune pada frame Benhil dan Gerbang Pemuda dengan kelas target resmi. Bila data belum cukup, gunakan dua tahap:

1. detector ringan menemukan semua objek;
2. classifier crop dijalankan sekali per track pada frame terbaik, bukan setiap frame.

Class voting dilakukan sepanjang umur track. Jika skor belum cukup, hasilnya `UNKNOWN`, tetapi objek tetap dihitung ke total lalu lintas. Jangan memaksa bus/truk ambigu menjadi kelas yang salah.

### P4 — Kendaraan cepat dan tracking

Kendaraan cepat terlewat karena gabungan: blur dari kamera, bbox kecil/jauh, detector lebih lambat dari input, frame sampling, dan ID tracker yang putus. Penanganannya:

- ukur waktu objek berada di ROI; target minimal 5 kesempatan detection per lintasan;
- detector adaptif 6–10 FPS, tracker berjalan pada setiap frame decode;
- gunakan Kalman prediction/optical-flow bridge di antara frame detection;
- garis hitung jangan terlalu dekat tepi frame;
- buat pre-count ROI agar track sudah matang sebelum menyentuh garis;
- crossing memakai segmen lintasan titik roda bawah, bukan bbox menyentuh garis;
- debounce per `(camera, track_id, line_id, direction)`;
- recover ID setelah occlusion pendek dengan IoU, arah, kecepatan, dan embedding ringan hanya jika perlu;
- evaluasi siang, malam, hujan, padat, dan kendaraan saling menutup.

Jika shutter kamera menghasilkan motion blur berat, software tidak dapat mengembalikan detail yang tidak pernah tertangkap. Namun hal itu baru boleh disebut limit sumber setelah blur rate dan recall pada frame mentah diukur.

### P5 — Counting canonical

Bedakan tiga angka yang sekarang mudah tercampur:

- `visible_now`: jumlah bbox valid pada frame;
- `unique_tracks`: jumlah ID unik selama jendela;
- `crossing_count`: kendaraan yang benar-benar melintasi garis.

Semua dashboard dan API wajib menyebut jenis metrik. `unique_tracks` tidak boleh ditampilkan sebagai jumlah kendaraan melintas. Total crossing:

```text
total = seluruh kelas resmi + ambiguous + unknown
```

Objek pengendara dan motornya harus di-dedup agar tidak menjadi motor + pejalan kaki. Line/ROI disimpan versioned karena perubahan geometri mengubah arti hitungan historis.

### P6 — Engine kemacetan

Kemacetan bukan satu hasil klasifikasi gambar. Gunakan time window 30–60 detik per lane/ROI:

```text
flow_vpm          = crossing kendaraan per menit
occupancy_ratio   = luas ROI yang ditempati bbox kendaraan
median_speed_px_s = median perpindahan track, dinormalisasi perspektif
queue_length      = panjang antrean/track lambat berurutan
stopped_ratio     = proporsi track di bawah ambang gerak
```

Status rule awal yang dapat dijelaskan:

```text
LANCAR  : speed tinggi, occupancy rendah
RAMAI   : occupancy naik, flow masih bergerak
PADAT   : occupancy tinggi, speed rendah
MACET   : occupancy tinggi + stopped_ratio tinggi selama N window
UNKNOWN : kualitas capture/tracking tidak memenuhi gate
```

Ambang harus berbeda per kamera/lajur dan dikalibrasi dari video berlabel. Simpan nilai metrik, versi rule, dan alasan verdict; jangan hanya menyimpan kata `MACET`.

### P7 — Pelanggaran

Aktifkan bertahap per kamera, bukan semua label sekaligus:

- **lawan arah:** arah track berlawanan dengan arah lane selama beberapa frame;
- **langgar marka:** jejak titik roda menyeberangi garis/polygon marka yang dikalibrasi;
- **parkir liar/ngetem:** track berada di forbidden ROI, kecepatan di bawah ambang, selama durasi minimum;
- **jalur sepeda:** kendaraan bermotor berada/melintas di polygon jalur sepeda sesuai rule.

Setiap event wajib membawa `camera_key`, `track_id`, jenis/kandidat kelas, waktu awal-akhir, bbox/trajectory, rule version, confidence, evidence frame/clip, dan status kualitas. Mode PoC saat ini menjeda siklus pelanggaran; itu harus diubah setelah pipeline tunggal siap, bukan menyalakan kembali pipeline ganda yang mahal.

## 6. Kontrak data minimal

```text
frame_observation
  camera_key, source_ts, received_at, frame_seq, frame_age_ms,
  decode_ms, dropped_before_infer, source_health

track_observation
  camera_key, track_id, ts, bbox, detector_label, detector_score,
  canonical_class, class_status, class_confidence,
  centroid, footpoint, velocity_px_s, direction, roi_ids

crossing_event
  camera_key, track_id, line_id, direction, crossed_at,
  canonical_class, class_status, rule_version, evidence_id

traffic_metric
  camera_key, roi_id, window_start, window_end,
  flow_vpm, occupancy_ratio, median_speed_px_s,
  queue_length, stopped_ratio, congestion_state,
  quality_score, rule_version

violation_event
  event_id, camera_key, track_id, kind, started_at, ended_at,
  rule_version, confidence, evidence_id, review_status
```

## 7. Observability yang wajib ada

Tanpa metrik ini, hardware tidak pernah bisa dinyatakan sebagai akar masalah:

- source FPS, decoded FPS, inferred FPS, tracked FPS;
- source uptime, reconnect count, decode error, frame gap;
- frame age p50/p95/p99;
- preprocess, inference, postprocess, tracker, rules, DB write p50/p95;
- queue depth, dropped-old-frame count;
- CPU per stage, RSS, throttling, context switch;
- detections/class, low-confidence rate, unknown rate;
- track fragmentation, ID switch, average track length;
- crossing count, duplicate suppression, evidence success;
- per-camera congestion inputs dan verdict.

Dashboard status harus dapat membedakan `SOURCE_DOWN`, `DEGRADED`, `INFERENCE_LAG`, `MODEL_LOW_CONFIDENCE`, dan `HEALTHY`.

## 8. Benchmark yang benar

Gunakan klip yang sama untuk semua eksperimen agar hasil dapat dibandingkan.

| Set | Kamera/kondisi | Durasi | Label manual minimum |
|---|---|---:|---|
| A | Benhil siang ramai | 15 menit | bbox, kelas, crossing, antrean |
| B | Benhil malam/hujan | 15 menit | bbox, kelas, crossing |
| C | Gerbang Pemuda siang cepat | 15 menit | bbox, kelas, crossing, arah |
| D | Gerbang Pemuda padat | 15 menit | crossing, queue, congestion, pelanggaran |

Setiap eksperimen menghasilkan CSV/JSON berisi konfigurasi, commit, model hash, jumlah core, resolusi, FPS sumber, metrik akurasi, latency, CPU, RAM, dan dropped frame.

Urutan run:

```text
E0 current baseline
E1 capture supervisor + bounded buffer
E2 single-pass pipeline, modul mahal on-demand
E3 ROI crop
E4 thread/NUMA sweep
E5 runtime comparison
E6 compact custom model FP32
E7 custom model INT8
E8 dua kamera simultan selama 2 jam
```

## 9. Kapan hardware boleh dinyatakan benar-benar mentok

Hardware hanya dinyatakan limit nyata jika **semua** kondisi berikut terbukti pada pipeline yang sudah disederhanakan:

1. sumber stabil dan decoded FPS memenuhi target;
2. satu decode dan satu detector dipakai, tanpa pipeline ganda;
3. ROI, bounded buffer, visualisasi off, batch DB, dan modul on-demand sudah aktif;
4. model compact yang memenuhi akurasi sudah diuji FP32 dan INT8/runtime alternatif;
5. thread dan NUMA sudah disapu, bukan hanya satu konfigurasi;
6. CPU jatah container bertahan di atas 85% selama minimal 30 menit;
7. p95 inference lebih besar dari interval frame target;
8. frame age atau dropped-frame rate tetap melampaui SLA;
9. RAM stabil, tidak ada leak/OOM, DB dan jaringan sudah dibuktikan bukan bottleneck;
10. akurasi tidak boleh diturunkan lagi untuk mengejar FPS.

Sebelum sepuluh bukti ini ada, pernyataan “CPU tidak kuat” belum valid.

## 10. Rencana eksekusi paling pendek

### Tahap A — Baseline dan reliability

- fokuskan konfigurasi hanya ke dua kamera;
- kumpulkan empat klip benchmark;
- pasang metrik per-stage;
- perbaiki reconnect capture;
- validasi garis dan ROI kedua kamera.

### Tahap B — Single-pass CPU

- bentuk `TrackObservation` canonical;
- satu detector untuk counting, kemacetan, dan pelanggaran;
- pindahkan OCR/atribut ke event-only;
- bounded buffer dan DB writer asinkron;
- benchmark thread/NUMA/runtime.

### Tahap C — Akurasi model

- label dataset dua kamera;
- ukur confusion matrix model sekarang;
- fine-tune compact model kelas Dishub;
- class voting per track dan `UNKNOWN` yang eksplisit;
- kalibrasi threshold per kelas, bukan satu threshold global.

### Tahap D — Analytics

- counting terarah dengan dedup;
- congestion window dan quality gate;
- rules pelanggaran per kamera;
- simpan evidence dan versi rule;
- endurance test dua kamera.

## 11. Acceptance gate PoC dua kamera

Angka dapat disepakati ulang dengan pemilik use case, tetapi tidak boleh dibiarkan tanpa target.

| Area | Gate awal |
|---|---|
| Capture | availability ≥99% selama 2 jam per kamera; reconnect otomatis |
| Freshness | p95 frame age ≤500 ms untuk RTSP/substream atau SLA HLS yang dicatat eksplisit |
| Detection | recall kendaraan ≥95% pada ground truth dua kamera; hasil per kelas dilaporkan |
| Fast vehicle | ≥5 observation opportunity di ROI dan miss rate ≤5% pada subset cepat |
| Classification | macro-F1 ≥90% untuk kelas yang disetujui; ambiguous/unknown tidak disembunyikan |
| Tracking | ID-switch ≤2% dan fragmentasi dilaporkan |
| Counting | error absolut ≤5% terhadap hitung manual, tanpa duplicate crossing |
| Congestion | macro-F1 state ≥90% terhadap label operator dan tidak memberi verdict saat quality gate gagal |
| Violation | precision ≥95%, recall ≥90%, setiap event memiliki evidence |
| Runtime | dua kamera serentak 2 jam, CPU container <85% rata-rata dan tidak ada backlog tak terbatas |

## 12. Yang tidak boleh dilakukan

- Jangan menaikkan core/container sebelum capture dan pipeline ganda dibereskan.
- Jangan mengklaim tujuh kelas dari model COCO yang tidak mempunyai kelas tersebut.
- Jangan menyebut `unique_tracks` sebagai jumlah crossing.
- Jangan menurunkan threshold sampai false positive meledak demi terlihat ramai.
- Jangan menjalankan OCR, atribut, dan lane segmentation untuk setiap objek/frame tanpa kebutuhan rule.
- Jangan menyimpan queue frame panjang; real-time membutuhkan frame terbaru, bukan backlog.
- Jangan memakai satu threshold kemacetan untuk dua sudut kamera yang berbeda.
- Jangan memberi verdict pelanggaran/kemacetan ketika source atau tracking berstatus buruk.
- Jangan melakukan overclock BIOS pada server bersama. Maksimalkan jalur software, affinity, dan jatah CPU yang aman terlebih dahulu.

## 13. Checkpoint implementasi 29 September 2026

Perubahan tahap pertama sudah dipasang dan diuji pada server CPU:

- antrean Live dikoreksi dari 80 menjadi dua frame terbaru;
- dropped frame, reconnect, effective FPS, inference p50/p95, dan queue delay sekarang terukur;
- bug kontrak `final_class` yang membuat setiap frame Live gagal diolah sudah diperbaiki;
- mode A/B `accurate` dan `compact` tersedia tanpa mengubah default dashboard;
- mode uji dua kamera tersedia dengan empat thread per kamera;
- sampling compact dinaikkan secara configurable (dua kamera diuji pada 0,33 detik);
- engine kemacetan provisional menghitung occupancy, median speed, stopped ratio, dan quality score;
- verdict kemacetan dipaksa `UNKNOWN` ketika track bergerak belum cukup;
- snapshot traffic terakhir disimpan di `runs/_latest/<camera>_traffic.json`;
- status dan tombol uji dapat dilihat di `/cpu-poc`.

Smoke test Gerbang Pemuda satu kamera:

| Mode | Inference | Queue p95 | Catatan |
|---|---:|---:|---|
| accurate / PP-YOLOE+ L | p95 559–674 ms | 1,51 dtk | akurasi baseline, terlalu lambat untuk target cepat |
| compact / PP-YOLOE S | p50 ±215 ms; p95 ±255 ms | ±0,63 dtk | sekitar 2,5× lebih cepat, akurasi wajib dibanding ground truth |

Smoke test dua kamera compact, empat thread per kamera:

| Kamera | Effective FPS awal | Infer p95 | Dropped frame awal | Pembacaan |
|---|---:|---:|---:|---|
| Benhil | ±3,5 FPS | ±384 ms | 0 | input relatif kontinu |
| Gerbang Pemuda | ±1,0–2,9 FPS | ±381 ms | 30 | HLS datang burst selain contention inference |

Kedua proses inference memakai sekitar 270% + 112% CPU pada snapshot proses, FFmpeg Benhil sekitar 59%, RAM container sekitar 1,04 GiB dari batas 5 GiB, dan host masih sekitar 90% idle. Artinya RAM bukan batas; seluruh 8 core container juga belum jenuh. Fokus berikutnya adalah menghilangkan dua proses model menjadi scheduler/inference tunggal, mengendalikan burst HLS, lalu benchmark thread/NUMA sebelum menambah jatah CPU.

### Checkpoint shared predictor dan burst HLS

Implementasi lanjutan 29 September 2026 sudah mengganti dua proses model
menjadi satu worker predictor dengan tracker yang terisolasi per kamera.
Proses yang terverifikasi saat aktif adalah tepat satu `_entry_langsung.py`
dan dua FFmpeg; saat Stop, ketiganya berhenti dan tidak meninggalkan proses
orphan.

Bug startup `LALIN_SOURCE` pada compatibility import sudah diperbaiki. Stderr
dan exit code worker sekarang masuk ke status API, sehingga proses yang mati
tidak lagi tampil sebagai `active`. Buffer input juga diubah dari dua frame
menjadi latest-window bounded maksimal enam detik. Perubahan ini diperlukan
karena HLS Gerbang Pemuda mengirim frame per-burst, berbeda dengan decode HD
Benhil yang relatif kontinu.

Hasil smoke test shared compact pada cadence 0,5 detik selama sekitar dua
menit:

| Kamera | Effective FPS | Infer p50 | Infer p95 | Received | Dropped | Reconnect |
|---|---:|---:|---:|---:|---:|---:|
| Benhil | 1,88 FPS | 230 ms | 271 ms | 248 | 6 | 0 |
| Gerbang Pemuda | 1,86 FPS | 222 ms | 264 ms | 240 | 19 | 0 |

Sebelum latest-window diperbesar, Gerbang Pemuda hanya sekitar 0,61 FPS dan
310 dari 385 frame input terbuang. Sesudah perubahan, distribusi inference
mendekati seimbang. Queue p95 masih 2,57 detik di Benhil dan 5,01 detik di
Gerbang karena pola burst HLS; metrik ini adalah usia sejak frame didecode,
bukan bukti freshness terhadap waktu kamera. Freshness end-to-end tetap perlu
timestamp sumber atau RTSP/substream sebelum dapat dinyatakan lulus.

Traffic verdict masih `UNKNOWN` secara sengaja karena quality score hanya
0,2-0,4 pada pengujian malam. Artinya sistem tidak mengarang status macet
ketika observasi gerak belum cukup. Akurasi counting, tujuh kelas, kemacetan,
dan pelanggaran tetap memerlukan ground truth/UAT sesuai acceptance gate.

### Rekaman siang dan dataset bootstrap

Dashboard existing `/demo` sekarang mempunyai source `Rekaman siang` untuk
`benhil2` dan `gerbangpemuda`. Source ini adalah replay terpisah; URL Live dan
konfigurasi dua kamera tidak diubah. Bila artifact baseline tersedia,
dashboard memutar video hasil tracking agar bbox dan track dapat diperiksa
langsung.

Dataset persiapan tersimpan di
`runs/training/daylight_v1` dan dibuat ulang secara deterministik oleh
`scripts/prepare_daylight_dataset.py`. Isinya 70 frame pada sampling 2 FPS:
56 train dan 14 validation. Tujuh kategori target sudah didefinisikan, tetapi
annotation COCO masih kosong dan manifest berstatus `ANNOTATION_REQUIRED`.
Dataset tidak boleh dipakai fine-tune sebelum bbox serta kelas direview.

Baseline tracking rekaman siang:

| Kamera | Frame | Track unik | Waktu proses |
|---|---:|---:|---:|
| Benhil CCTV-2 | 152 | 37 | 35,9 dtk |
| Gerbang Pemuda | 200 | 28 | 31,1 dtk |

Registry job juga diperbaiki agar dua job pada detik yang sama mendapat ID
unik dan tidak menulis ke run directory yang sama.

## 14. Kesimpulan

Bottleneck paling awal saat ini adalah **sumber/capture yang tidak stabil**. Sesudah frame tersedia, bottleneck komputasi terbesar adalah **detector sekitar 124–125 ms/frame**, lalu modul atribut/pelat yang menurunkan pipeline Gerbang Pemuda menjadi sekitar 4,5–5,5 FPS. Namun target tujuh kelas, kemacetan, dan pelanggaran belum selesai hanya dengan menambah FPS: **taksonomi model belum mencukupi, engine kemacetan belum ada, dan rule pelanggaran belum tervalidasi**. Jalan tercepat untuk dua kamera adalah capture supervisor, pipeline tunggal, detector compact khusus kelas Dishub, tracker stabil, kemudian counting/congestion/violation memakai track canonical yang sama. Baru setelah semua optimasi dan gate Bagian 9 gagal, CPU boleh dinyatakan sebagai limitasi nyata.
