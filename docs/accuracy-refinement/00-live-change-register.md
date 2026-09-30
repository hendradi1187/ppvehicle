# 00 — Register Perubahan Live & Baseline v0

Dokumen ini mencatat setiap perubahan yang **sudah aktif di server produksi**
(`100.96.136.65:8100`, kontainer `docker-ppvehicle-1`) yang memengaruhi
deteksi, klasifikasi, atau penghitungan, sejak proyek penyempurnaan akurasi
dimulai. Tujuannya: perbandingan model nanti dilakukan terhadap baseline yang
**tetap**, dan setiap perubahan bisa dikembalikan.

> **Catatan versi kode.** Proyek ini **bukan repositori git**, jadi tidak ada
> nomor *commit*. Sebagai penggantinya dipakai **sidik jari SHA-256** tiap
> berkas yang aktif. Disarankan `git init` lokal (tanpa remote) sebelum
> implementasi berikutnya supaya setiap perubahan punya commit — menunggu
> persetujuan pemilik proyek.

## Baseline v0 — DIBEKUKAN

| Item | Nilai |
|---|---|
| ID | `baseline-v0` |
| Dibekukan | 26-09-2026, setelah LC-005 (berkas terakhir dipasang 08:27:18 WIB) |
| Salinan berkas | `docs/accuracy-refinement/baseline-v0/berkas/` |
| Manifest | `docs/accuracy-refinement/baseline-v0/MANIFEST.sha256` |
| Verifikasi | SHA-256 berkas di server = salinan lokal = salinan baseline (dicek 26-09-2026) |
| Detektor | PP-YOLOE+ L COCO (`/models/ppyoloe_plus_l_coco`, input 640×640 statis) |
| Tracker | JDETracker/ByteTrack (`configs/tracker_hitung.yml`, sha `f0f9c519…`) |
| Konfigurasi garis | `configs/garis_hitung.yml` (sha `b566a063…`) |

**Aturan:** mulai sekarang setiap perubahan yang memengaruhi keluaran
deteksi/kelas/hitungan WAJIB mendapat ID `LC-xxx` baru di tabel di bawah dan
dievaluasi terhadap `baseline-v0` dengan skrip evaluasi yang sama. Baseline
tidak boleh diubah diam-diam selama eksperimen.

Sidik jari berkas baseline-v0 (16 karakter pertama; lengkapnya di MANIFEST):

| Berkas | SHA-256 |
|---|---|
| `src/lalin/langsung.py` | `00936b3ea28893ba` |
| `src/lalin/_entry_langsung.py` | `8ffeea0394b802b3` |
| `src/lalin/_entry_hitung.py` | `ea2b5def6e6ec87e` |
| `src/lalin/api.py` | `249b8cf16d4f695a` |
| `src/lalin/batch.py` | `50c272b77bccbcda` |
| `src/lalin/runner.py` | `8dfa209e187a8c83` |
| `src/lalin/marka_putih.py` | `e75e081af55ba9dc` |
| `src/lalin/settings.py` | `00a17644a8fdba4b` |
| `src/lalin/web/assets/bbox.js` | `ab8c674cc6c6ae68` |
| `src/lalin/web/dashboard.html` | `c4bfcba52ee7377d` |
| `configs/garis_hitung.yml` | `b566a063caed6ef9` |
| `configs/tracker_hitung.yml` | `f0f9c5197dbade13` |
| `configs/cameras.yml` | `b09537353cd23f38` |

## Register perubahan

Waktu dalam WIB. "Dipasang" = waktu berkas terakhir ditulis di server
(`stat` mtime) bila tersedia; selain itu perkiraan dari log sesi.

### LC-001 — Kotak deteksi di peramban (COCO-SSD lite)
- **Tanggal:** 25-09-2026 ±20.40
- **Berkas:** `web/assets/bbox.js` (v1), `web/assets/tfjs/*` (pustaka + bobot `ssdlite_mobilenet_v2`), `web/dashboard.html`, `web/assets/fonts/*`
- **Perilaku:** model mini di peramban memberi kotak 7 kelas pada video Live.
- **Status:** **DIGANTIKAN** oleh LC-003 (model ini salah membaca pesepeda sebagai pejalan kaki). Berkas `tfjs/` masih ada di server tetapi tidak dimuat.
- **Rollback:** tidak relevan.

### LC-002 — Garis hitung per kamera + pencocokan posisi PTZ
- **Tanggal:** 25-09-2026 ±21.05–22.00
- **Berkas:** `configs/garis_hitung.yml`, `src/lalin/batch.py` (load/simpan tampilan), `src/lalin/api.py` (`/api/garis-hitung/{key}`), `web/assets/bbox.js`, `web/dashboard.html`
- **Perilaku:** kendaraan dihitung saat titik roda melintasi ruas garis per kamera; syarat bergerak ≥14 px (`GERAK_MIN_PX`); garis disimpan per posisi kamera (tampilan) dengan acuan gambar 80×36 abu-abu.
- **Rollback:** tidak ada salinan pra-perubahan penuh (bukan git). Menonaktifkan hitungan: kosongkan `configs/garis_hitung.yml` → kamera tanpa garis tidak menghitung.

### LC-003 — Deteksi langsung di server, sinkron PTS
- **Tanggal:** 25-09-2026 ±22.30 s.d. 26-09-2026 dini hari; `runner.py` (nice 10) 25-09 21:27
- **Berkas:** `src/lalin/_entry_langsung.py` (baru), `src/lalin/langsung.py` (baru), `src/lalin/api.py` (`/api/langsung*`), `src/lalin/runner.py` (`_prioritas_rendah`), `web/assets/bbox.js` (v2: gambar hasil server), `web/dashboard.html` (Live tidak lagi menjalankan Pantau/HUD)
- **Perilaku:** PP-YOLOE+ L + ByteTrack pada aliran HLS (`ffmpeg -copyts`, 1 frame/0,6 dtk, 6 thread); kotak ditampilkan pada frame video dengan PTS sama, diinterpolasi. Siklus otomatis & Pantau berjalan pada nice 10.
- **Rollback:** tidak ada salinan pra-perubahan penuh. Mematikan: `POST /api/langsung/henti`; Live kembali tanpa kotak.

### LC-004 — Penyunting garis presisi, acuan siang/malam
- **Tanggal:** 26-09-2026 07.43–08.10 (`marka_putih.py` 07:43:58, `dashboard.html` 07:48:02, `batch.py` 08:06:37, `garis_hitung.yml` 08:07:30)
- **Berkas:** `src/lalin/marka_putih.py` (baru, `/api/garis-putih/{key}`), `src/lalin/batch.py` (`acuan_2`), `src/lalin/langsung.py` (acuan dinamis), `web/assets/bbox.js`, `web/dashboard.html`, `configs/garis_hitung.yml` (garis 6 kamera; garis Gerbang Pemuda digambar pemilik proyek 25-09 21.45 di atas garis henti)
- **Perilaku:** tidak mengubah deteksi/kelas; mengubah **di mana** kendaraan dihitung dan **kapan** hitungan dijeda (kamera bergeser).
- **Rollback:** garis: salinan di `baseline-v0/berkas/configs/garis_hitung.yml` adalah versi saat ini; versi sebelumnya tidak tersimpan.

### LC-005a — Batas ukuran kotak siang & evaluasi siang/malam tiap menit
- **Tanggal:** 26-09-2026 ±08.12
- **Berkas:** `src/lalin/_entry_langsung.py` (`wajar_langsung`, `keadaan`)
- **Perilaku:** siang hari kotak sampai 60% frame diterima (bus Transjakarta tepat di bawah kamera sebelumnya dibuang oleh batas 20%); malam tetap 20%. Status siang/malam dan ambang dievaluasi ulang tiap 60 dtk.
- **Rollback:** tidak ada salinan pra-perubahan penuh (dipasang lewat skrip sekali-pakai).

### LC-005 — Ambang yakin per kelas, "Tidak Dikenal", truk tidak dipecah, masker OSD
- **Tanggal:** 26-09-2026 08:27:16–08:27:18 WIB (menanggapi UAT: gerobak dibaca "Truk Berat 55%")
- **Berkas:** `src/lalin/langsung.py`, `src/lalin/_entry_langsung.py`, `src/lalin/api.py`, `web/assets/bbox.js`
- **Konfigurasi (tertanam di kode, belum di berkas konfigurasi pusat):** `YAKIN = {sepeda_motor .45, mobil_penumpang .50, bus_besar .60, truk .60, sepeda .45, pejalan_kaki .50}`, `YAKIN_PORSI = 0.5`, `TAHAN_DTK = 6.0`, pita OSD `y2 ≤ 34 atau cy ≤ 22 atau (cy ≥ 316 dan x1 < 340)`.
- **Perilaku saat ini:** sebuah track diberi kelas hanya bila skornya ≥ ambang pada ≥50% pengamatannya; selain itu `tidak_dikenal`. Truk COCO → `truk` ("Truk (Sedang/Berat)"), tidak dipecah Kendaraan Sedang/Truk Berat. Kotak di pita teks OSD dibuang. Peristiwa lintas garis ditahan sampai kelas track final, lalu dikirim dengan nomor urut (`seq`, parameter `sejak_p`).
- **Dasar angka:** pemeriksaan visual 104 kotak truk/bus dari 6 kamera (26-09 pagi). Ambang kelas lain **belum dikalibrasi** dengan data berlabel.
- **Rollback (tersedia):** versi pra-LC-005 direkonstruksi dengan membalik pasangan teks lama→baru dari skrip patch pemasangan, di `docs/accuracy-refinement/baseline-v0/pra-LC005/` (`langsung.py`, `_entry_langsung.py`, `api.py`, `bbox.js`; lolos cek sintaks). Langkah:
  ```bash
  cd /d/APPS/ppvehicle/docs/accuracy-refinement/baseline-v0/pra-LC005
  scp src/lalin/langsung.py src/lalin/_entry_langsung.py src/lalin/api.py user3@100.96.136.65:/home/user3/ppvehicle/src/lalin/
  scp src/lalin/web/assets/bbox.js user3@100.96.136.65:/home/user3/ppvehicle/src/lalin/web/assets/
  ssh user3@100.96.136.65 "docker restart docker-ppvehicle-1"
  ```
  Hanya kontainer `docker-ppvehicle-1` yang di-restart. Kembali ke v0: salin dari `baseline-v0/berkas/` dengan cara yang sama.

### LC-006 — Perbaikan cacat LC-005: masker OSD presisi, penanda sesi, cegah proses ulang
- **Tanggal:** 26-09-2026 09:19:35 WIB (atas perintah pemilik proyek: "perbaiki sekarang saja")
- **Dibandingkan terhadap:** `baseline-v0` (sidik jari server sebelum pasang = v0: `8ffeea03…`, `00936b3e…`, `ab8c674c…`)
- **Berkas & sidik jari sesudah:** `src/lalin/_entry_langsung.py` `7f378067a8d8c1fc`, `src/lalin/langsung.py` `e45f6080323d8869`, `web/assets/bbox.js` `409f755fc75b4f31`. Salinan + manifest: `docs/accuracy-refinement/lc-006/`; skrip patch `lc-006/patch_lc006.py`; uji simulasi `lc-006/uji_bbox_sesi.cjs`.
- **Cacat yang diperbaiki:**
  1. *(K1)* Masker OSD LC-005 membuang pita y ≤ 34 dan seluruh kiri-bawah hingga x 340: dari 55 kotak yang dibuang pada 821 deteksi uji, hanya 11 teks; 40 kendaraan nyata ikut hilang; pita bawah menutup 24% garis hitung Gerbang Pemuda (x 217–272). **Sekarang:** kotak teks diukur dari latar median 7 kamera (atas y 2–11 selebar frame; kiri bawah y 322–331, x 0–245, sama di ketujuh kamera) → `OSD_KOTAK = ((0,0,640,14),(0,318,252,335))`; kotak dibuang hanya bila tinggi ≤ 22 px, lebar ≥ 1,2 × tinggi, dan ≥ 60% luasnya di dalam kotak teks. Pada data yang sama: 12 dibuang, semuanya teks (diperiksa visual); 43 objek nyata kembali.
  2. *(C5)* Nomor urut peristiwa (`seq`) mulai lagi dari 0 setiap sesi server, sedangkan peramban tetap meminta `sejak_p` lama → peristiwa sesi baru tersaring diam-diam. **Sekarang:** server mengirim `sesi` (id sesi); peramban yang melihat sesi berganti me-reset `sejak`/`sejak_p`/hasil frame lalu meminta ulang. Dibuktikan dengan simulasi respons (`uji_bbox_sesi.cjs`): kode lama tetap meminta `sejak_p=3` setelah sesi baru; LC-006 meminta `sejak_p=0` lalu lanjut.
  3. *(D2)* Setelah ffmpeg tersambung ulang (`-live_start_index -2`), ±15 dtk yang sudah dianalisis diproses ulang oleh tracker yang tetap hidup → kendaraan sama bisa terhitung dua kali. **Sekarang:** frame dengan PTS tidak maju dibuang (mundur > 600 dtk dianggap putaran/diskontinuitas dan diterima).
- **Tidak diubah:** ambang, YAKIN, pemetaan kelas, tracker, garis hitung, API lama (hanya field tambahan `sesi`).
- **Rollback:** salin `baseline-v0/berkas/src/lalin/{_entry_langsung.py,langsung.py}` dan `baseline-v0/berkas/src/lalin/web/assets/bbox.js` ke server, lalu `docker restart docker-ppvehicle-1`.
- **Belum terukur:** dampak pada jumlah hitungan per kamera (butuh ground truth — lihat rencana evaluasi).

### LC-007 — Lapisan pemetaan kelas pusat (isi kelas mengikuti PKJI 2023)
- **Tanggal:** 26-09-2026 16:06:28 WIB (keputusan pemilik proyek 26-09 siang: "Ikut aturan yang ada saja segera sesuaikan … di codebase")
- **Dibandingkan terhadap:** LC-006 (sidik jari server sebelum pasang = LC-006: `e45f6080…`, `409f755f…`)
- **Berkas baru:** `configs/klasifikasi.yml` `973871efa6889fa6` (satu-satunya sumber kelas: 7 nama kelas surat 1316 = OFFICIAL; isi kelas PKJI 2023 = PROVISIONAL, `approved_by` kosong; L2 jenis rinci → L3; L1 label detektor → keputusan; ambang yakin), `src/lalin/klasifikasi.py` `181cec9076e3d7f0` (pemuat + validasi + `putuskan()`), `tests/test_klasifikasi.py` (11 uji, lolos di Python 3.10 kontainer pada salinan sementara).
- **Berkas diubah:** `src/lalin/langsung.py` `2df9ec3ef68fb430` (KE_KELAS/YAKIN/kelas_track dihapus; aturan dibaca tiap sesi Live), `web/assets/bbox.js` `68fb725f055bfb39` (nama/warna dari server; panel 7 kelas surat + "Bus (belum terpilah)" + "Truk (belum terpilah)" + Tidak Dikenal). Salinan + manifest + skrip patch: `docs/accuracy-refinement/lc-007/`.
- **Perubahan perilaku:**
  - COCO `bus` → **AMBIGUOUS {Kendaraan Sedang, Bus Besar}**, kunci `bus` (sebelumnya langsung `bus_besar`). Alasan: menurut PKJI 2023 Metromini/bus sedang = Kendaraan Sedang; COCO tidak membedakan.
  - COCO `truck` → **AMBIGUOUS {Mobil Penumpang, Kendaraan Sedang, Truk Berat}**, kunci `truk` (pikap = MP menurut PKJI).
  - `motorcycle`, `car`, `person`, `bicycle` → CLASSIFIED seperti sebelumnya. Ambang yakin & porsi 0,5 **tidak berubah** (diuji).
  - Peristiwa lintas mendapat field tambahan: `status`, `final_class` (null bila AMBIGUOUS/UNKNOWN), `kandidat`, `label_detektor`, `n_lihat`, `n_yakin`, `versi_aturan`. Kunci `kelas` lama tetap. Respons `/api/langsung` mendapat `klasifikasi` (versi + peta tampil).
- **Belum diubah:** mesin hitung7/Pantau (`_entry_hitung.py`) masih memakai pemetaan lama (catatan audit 01 L2).
- **Rollback:** salin `lc-006/berkas/src/lalin/{langsung.py}` dan `lc-006/berkas/src/lalin/web/assets/bbox.js` ke server, hapus/biarkan `configs/klasifikasi.yml` dan `src/lalin/klasifikasi.py` (tidak dipakai lagi), `docker restart docker-ppvehicle-1`.

### LC-008 — Garis hitung CCTV-01 dipindah dari y 290 ke y 145 (berbasis data)
- **Tanggal:** 26-09-2026 ±16:12 WIB, lewat `PUT /api/garis-hitung/cctv01` (indeks 0; acuan posisi dipertahankan).
- **Masalah:** 0 peristiwa lintas dalam > 3 menit lalu lintas ramai. Di CCTV-01 kendaraan bergerak menjauhi kamera dan masuk dari tepi bawah; garis y 290 berada di titik masuk, sehingga sebagian besar kendaraan baru terdeteksi setelah melewatinya (roda p90 = y 160; median panjang track 3 frame).
- **Dasar pemilihan:** dari 397 frame Live, jumlah lintasan antar-dua-pengamatan-berurutan per tinggi garis: y 290 → 2; y 200 → 5; y 150 → 17; y 110 → 31. Dipilih y 145 (kompromi: cukup banyak lintasan, objek belum terlalu kecil).
- **Baru:** `[[205,145,560,145]]` (tepi kiri jalur sepeda s.d. bahu dekat halte). Hasil: 6 peristiwa dalam 139 dtk (2 MP, 1 SM, 2 Bus belum terpilah, 1 Truk belum terpilah).
- **Rollback:** salinan konfigurasi sebelum perubahan: `docs/accuracy-refinement/lc-006/garis_hitung.pra-LC008.yml`; atau `PUT /api/garis-hitung/cctv01` dengan `{"garis":[[191,290,609,290]],"indeks":0}`.
- **Tindak lanjut:** posisi garis kamera lain belum divalidasi dengan data yang sama.

### LC-009 — Mode POC (fokus butir 1–2 surat 1316) + satu panel penghitungan
- **Tanggal:** 26-09-2026 22:13:13 WIB (pasang), 22:19 WIB (penanda versi skrip). Atas arahan pemilik proyek: hanya garis hitung di video, satu panel hitungan, siklus pelanggaran dijeda, fitur lain disembunyikan lewat Mode POC.
- **Dibandingkan terhadap:** LC-007/LC-008 (sidik jari server sebelum pasang: `api.py` `249b8cf1…`, `dashboard.html` `c4bfcba5…`, `bbox.js` `68fb725f…`; salinan di `lc-009/pra/`).
- **Berkas & sidik jari sesudah:** `src/lalin/api.py` `4678313b08f233d2`, `web/dashboard.html` `2887960e54881721`, `web/assets/bbox.js` `c3ce8704e078a03e`, berkas baru `configs/mode_poc.yml` `2868aa0c4e65f355`. Server = lokal (diperiksa SHA-256). Salinan + manifest + skrip patch: `docs/accuracy-refinement/lc-009/`.
- **Perubahan perilaku:**
  - `configs/mode_poc.yml` (`aktif`, `jeda_siklus`) + `GET/PUT /api/mode-poc`. Saat `jeda_siklus` benar, worker siklus pelanggaran otomatis dihentikan dan tidak dinyalakan saat start-up. Tombol "Mode POC" di kepala dasbor; mematikannya minta konfirmasi, lalu siklus jalan lagi.
  - Dengan Mode POC nyala, dasbor menyembunyikan (tidak menghapus): kolom kanan, panel AI/hitung/kualitas, grafik, KPI, verifikasi, aduan, lonceng, lapisan zona, tab selain Live, readout pelanggaran, dan lencana "Perlu tindakan" (diganti "Live"). Video hanya menampilkan kotak deteksi + garis hitung.
  - Panel di bawah video ditulis ulang: "PENGHITUNGAN LALU LINTAS", kartu 7 kelas resmi (KS/BB/TB diredupkan "belum terpilah"), baris "Belum terklasifikasi" (Bus/Truk belum terpilah, Tidak Dikenal), hitungan per arah per garis. Peringatan yang menentukan keandalan angka (belum ada garis, kamera bergeser, tanpa acuan posisi, malam, galat) selalu tampil; rincian mesin dan SMP di "Detail teknis" yang dapat dilipat (status buka dipertahankan saat panel digambar ulang).
  - `<script src="/aset/bbox.js?v=lc009">`: `/aset/*` tidak mengirim Cache-Control, sehingga Chrome sempat memakai `bbox.js` lama dari cache. Naikkan penanda `v=` setiap kali `bbox.js` berubah.
- **Tidak diubah:** mesin deteksi/tracker, ambang, pemetaan kelas, garis hitung, data DB, fitur pelanggaran (hanya disembunyikan/dijeda).
- **Diverifikasi (Chrome, 22:20 WIB):** `/api/mode-poc` → `{"aktif":true,"jeda_siklus":true,"siklus_berjalan":false}`; kolom kanan, readout, dan skor kamera `display:none`; video Gerbang Pemuda hanya menampilkan garis hitung; panel baru tampil; "Detail teknis" tetap terbuka setelah digambar ulang.
- **Rollback:** tombol "Mode POC" → mati (tanpa ganti berkas), atau salin `lc-009/pra/src/lalin/{api.py,web/dashboard.html,web/assets/bbox.js}` ke server, hapus `configs/mode_poc.yml`, lalu `docker restart docker-ppvehicle-1`.

### LC-010 — Mode POC: rel hitungan di kanan video (usulan A+B)
- **Tanggal:** 26-09-2026 22:27:41 WIB (pasang), 22:31 (baris kelas diringkas), 22:36 (getter diagnostik). Atas pilihan pemilik proyek: "A+B dulu, baru C+D".
- **Dibandingkan terhadap:** LC-009 (server sebelum pasang: `dashboard.html` `2887960e…`, `bbox.js` `c3ce8704…`; salinan di `lc-010/pra/`).
- **Berkas & sidik jari sesudah:** `web/dashboard.html` `50f40733078edbe9`, `web/assets/bbox.js` `21a32a4a256bcce1`. Server = lokal. Salinan + manifest + skrip patch: `docs/accuracy-refinement/lc-010/`.
- **Perubahan perilaku (hanya tampilan, hanya saat Mode POC nyala):**
  - Lebar ≥ 1100 px: kartu video dibagi dua kolom, video di kiri, `#bbhud` menjadi rel 300 px di kanan. Tinggi rel mengikuti kolom video (`contain: size`); isi berlebih digulir dengan scrollbar tipis. Sebelum Live berjalan (rel `hidden`), video memakai lebar penuh (`:has(#bbhud[hidden])`). Daftar kamera 278 → 236 px. Lebar < 1100 px: tetap bertumpuk seperti LC-009.
  - Rel: total melintas ditulis besar; peringatan dalam kotak berwarna; 7 kelas satu baris per kelas (nama · "tampak n"/"belum terpilah" · angka); "Belum terklasifikasi" bertumpuk.
  - Toolbar "Tampilan" disembunyikan di Mode POC (hanya berisi tombol Live).
  - `bbox.js`: angka total dibungkus `<b class="bb-n">` (isi sama); `BBox.diag` getter hanya-baca untuk diagnosis PTS/peristiwa dari konsol. Penanda skrip `?v=lc010b`.
- **Tidak diubah:** logika hitung, sinkron PTS, kelas, ambang, server.
- **Diverifikasi (Chrome 1482 px):** video 699×393, rel 300×503, isi rel 552 px (7 kelas + 2 baris belum terklasifikasi terlihat tanpa gulir). Hitungan 0 selama uji dijelaskan `BBox.diag`: tab Chrome tersembunyi (video beku, `mediaTime` tetap) dan server hanya mencatat 1 lintasan dalam ±200 dtk (PTS 8141, 2 dtk sebelum panel mulai `mulaiPts` 8143) — bukan cacat LC-010.
- **Belum diverifikasi:** tampilan < 1100 px di Chrome (jendela tidak dapat diperkecil dari alat), hitungan bertambah pada tab terlihat setelah LC-010.
- **Rollback:** salin `lc-010/pra/src/lalin/web/{dashboard.html,assets/bbox.js}` ke server (tanpa restart; berkas statis).

### LC-011 — Tren per menit, log lintasan, Unduh CSV, Layar TV (usulan C+D)
- **Tanggal:** 26-09-2026 22:42:31 WIB (pasang), 22:54 (label sumbu), 22:56 (CSS Layar TV). Atas perintah pemilik proyek: "lanjutkan!".
- **Dibandingkan terhadap:** LC-010 (server sebelum pasang: `dashboard.html` `50f40733…`, `bbox.js` `21a32a4a…`; salinan di `lc-011/pra/`).
- **Berkas & sidik jari sesudah:** `web/dashboard.html` `242e0bba01dd0bfb`, `web/assets/bbox.js` `d844ea19955678f2`. Server = lokal. Salinan + manifest + skrip patch: `docs/accuracy-refinement/lc-011/`.
- **Perubahan perilaku:**
  - Panel baru "Tren & log lintasan" di bawah kartu video: grafik batang bertumpuk per kelas per menit (30 menit terakhir; rata-rata/menit, menit tersibuk, skala) dan 12 lintasan terakhir (jam tampil, kelas, status dengan kandidat bila AMBIGUOUS, garis & arah, observasi yakin/total).
  - **Unduh CSV**: semua lintasan sesi Live, dibuat di peramban dari catatan yang sama dengan panel (pemisah `;`, UTF-8 dengan BOM untuk Excel berlokal Indonesia). Tidak mengambil dari DB.
  - **Layar TV**: `body.mode-tv` + layar penuh; hanya video + rel hitungan (rel 340 px, huruf besar), lebar video dibatasi tinggi layar. Keluar dengan Esc atau tombol "Keluar layar TV".
  - `bbox.js`: lintasan yang sudah terputar dicatat **sekali** (`tercatat`, kunci sesi server + seq) dengan jam tampil. Hitungan rel, tren, log, dan CSV semuanya dibaca dari catatan itu. Catatan diarsipkan saat sesi browser berhenti dan dipulihkan bila kamera yang sama dimulai lagi (video tersambung ulang / tab sempat tersembunyi), sehingga angka tidak kembali ke 0. Muat ulang halaman atau ganti kamera tetap mulai dari 0.
- **Diverifikasi (Chrome):** 1 lintasan di rel = di log = di DB (22.52); CSV 9 baris dengan kolom sesuai kepala (isi ditangkap di halaman, tidak diunduh ke disk); Layar TV 1536×808: video 1156×650, rel 340×760, angka 46 px; Esc → normal, Mode POC tetap nyala. Tidak ada galat konsol.
- **Rollback:** salin `lc-011/pra/src/lalin/web/{dashboard.html,assets/bbox.js}` ke server (tanpa restart).

### LC-012 — Lintasan & cakupan penghitungan disimpan ke PostgreSQL
- **Tanggal:** 26-09-2026 22:48:08 WIB (DDL), 22:49:08 WIB (kode + restart `docker-ppvehicle-1`). Atas perintah pemilik proyek: "kan seharusnya log tersimpan di database!" — mengubah keputusan sebelumnya ("kita tunggu saja") yang menunda penyimpanan sampai taksonomi dikunci.
- **Mengapa aman sebelum taksonomi dikunci:** yang disimpan adalah data mentah per kendaraan (label detektor, status, kandidat, jumlah pengamatan, skor) + `versi_aturan`. Bila isi kelas berubah setelah konfirmasi Dishub, hitungan dapat dihitung ulang dari kolom mentah.
- **DB (hanya aditif, sebagai `lalin_app`):** `lalin.sesi_hitung`, `lalin.lintas_kendaraan`, view `lalin.v_hitung_15mnt` (DDL `docs/ddl/lintas_kendaraan.sql`, juga ditambahkan ke `docs/ddl/skema_postgres.sql` dan `docker/pg-init/skema.sql.inc` untuk volume baru). Tabel lama tidak disentuh. `lalin_baca` (tim DB) dapat membaca (diperiksa).
- **Uji sebelum pasang:** 6 uji unit `tests/test_simpan_lintas.py` (koneksi tiruan) lolos di Python 3.10 kontainer pada salinan `/tmp`; `lc-012/uji_pg_rollback.py` menjalankan DDL + upsert sesi + insert lintas + duplikat + view + CHECK status di PostgreSQL sungguhan **dalam satu transaksi yang di-ROLLBACK**; sesudahnya daftar tabel skema `lalin` sama dengan sebelum uji (tidak ada data uji di DB produksi).
- **Berkas & sidik jari sesudah:** `src/lalin/langsung.py` `2e056d629b2fbe15`, baru `src/lalin/simpan_lintas.py` `659d8c5f5c44a54d`, `docs/ddl/lintas_kendaraan.sql` `effaae5c4378efca`, `docker/pg-init/skema.sql.inc` = `docs/ddl/skema_postgres.sql` `d1b6ae59c00c0fe5`. Pra: `langsung.py` `2df9ec3e…`, `skema.sql.inc` `71556663…` (`lc-012/pra/`). Salinan + manifest + skrip patch + skrip uji: `docs/accuracy-refinement/lc-012/`.
- **Perilaku:** penulisan di utas `pencatat-lintas` dengan antrean (maks 5000); DB lambat/mati tidak menghambat deteksi — data ditahan dan dicoba lagi tiap 30 dtk; status di `/api/langsung` → `db` (`tertulis`, `antre`, `dibuang`, `galat`). Sesi dicatat saat mulai, diperbarui tiap 30 dtk dan saat berhenti (`n_frame`, `detik_terhitung` = detik dengan garis aktif, `n_lintas`, `catatan` alasan berhenti). Peristiwa `/api/langsung` mendapat field tambahan `t_server` dan `bbox`.
- **Diverifikasi:** lintasan yang ditemukan analisis offline dari data frame (mobil, PTS 9199) tercatat server dan masuk `lintas_kendaraan` (22.51.56 WIB, CLASSIFIED, `car`, 4/3 pengamatan, skor maks 0,701, malam); `sesi_hitung` terisi (295 frame, 186,3 dtk terhitung); dibaca dengan akun `lalin_baca`.
- **Rollback kode:** salin `lc-012/pra/src/lalin/langsung.py` ke server, hapus `src/lalin/simpan_lintas.py`, `docker restart docker-ppvehicle-1` (penulisan berhenti). Tabel dibiarkan (berisi data nyata); menghapusnya hanya dengan persetujuan pemilik proyek dan tim DB.
- **Keterbatasan (LC-012):** hanya kamera yang sedang ditonton di Live yang dihitung (1 kamera sekaligus, berhenti 60 dtk setelah tidak ada penonton) — lihat `sesi_hitung` untuk cakupan. `waktu` = jam server saat frame diolah (tertinggal latensi HLS beberapa detik dari kejadian). Unduh CSV masih dari peramban, belum dari DB.

### LC-013 — Mode POC: daftar kamera pindah ke bawah panel tren
- **Tanggal:** 26-09-2026 23:00:57 WIB (pasang), 23:02 (baris wilayah kosong melebar), 23:03 (gulir langsung). Atas usulan pemilik proyek: "Daftar kamera di pindahkan" ke bawah data tren supaya tidak ada ruang kosong.
- **Dibandingkan terhadap:** LC-011 (server sebelum pasang: `dashboard.html` `242e0bba…`; salinan di `lc-013/pra/`).
- **Berkas & sidik jari sesudah:** `web/dashboard.html` `86fb34a30a28ea6c`. Server = lokal. Salinan + manifest + skrip patch: `docs/accuracy-refinement/lc-013/`.
- **Perubahan perilaku (hanya saat Mode POC nyala):**
  - Satu kolom: video + rel (atas) → tren & log → daftar kamera (bawah).
  - Lebar video = min((tinggi layar − 230 px) × 16/9, lebar − 300 px), sehingga video + pemutar muat tanpa gulir; sisa lebar untuk rel (≥ 300 px).
  - Daftar kamera sebagai kisi kartu per wilayah (≥ 250 px per kartu); wilayah tanpa kamera diringkas satu baris beserta catatannya. Pencarian + chip wilayah satu baris.
  - Klik kartu kamera → halaman kembali ke atas (ke video). Gulir langsung, bukan animasi: animasi gulir tidak berjalan saat tab Chrome tersembunyi (terukur).
  - `renderCams()` membungkus tiap wilayah dalam `div.wilgrp` / `div.wilkosong`; tampilan non-POC tetap bertumpuk seperti sebelumnya.
- **Diverifikasi (Chrome 1482×711):** video 855×481 + pemutar (bawah 708 px) di dalam layar; rel 391×591 tanpa gulir; tren 1248 px; daftar kamera 7 kartu + 4 wilayah kosong; klik kartu → `scrollY` 763 → 0, video & rel tetap jalan; tidak ada galat konsol.
- **Rollback:** salin `lc-013/pra/src/lalin/web/dashboard.html` ke server (tanpa restart).

### LC-014 — Garis hitung Gerbang Pemuda mencakup semua lajur + bus/truk satu pelacak + perbaikan acuan kedua
- **Tanggal:** 26-09-2026 23:13:33 WIB (kode + restart), 23:16:35 (garis), 23:17:47 (perbaikan `batch.py` + restart). Atas perintah pemilik proyek: "mulai dari (1) dan (2)".
- **Masalah (diukur pada data frame Live 22.59–23.09):**
  1. Garis `[[35,175,272,359]]` menempel garis henti di zebra cross dan berakhir di tepi bawah pada x 272: lajur kanan tidak tertutup. Kendaraan di ruas ini bergerak MENUJU kamera, sehingga melintasi garis henti saat paling dekat — paling besar dan paling kabur di malam hari (jarang terdeteksi). Hasil: 2 lintasan dalam 7,4 menit, keduanya pejalan kaki.
  2. Pelacak MCMOT memberi ID per kelas COCO: kendaraan besar yang labelnya berganti bus ↔ truck antar-frame terpecah menjadi banyak track pendek yang tidak pernah terhitung (contoh: satu MPV tosca yang berhenti di bawah kamera, salah dibaca bus/truk, menjadi 15 track dalam ±2 menit).
  3. Ditemukan saat mengganti garis: `simpan_garis_hitung` membuang `acuan_2` (acuan malam) setiap kali garis diganti tanpa acuan baru. Acuan malam Gerbang Pemuda sempat hilang dan dipulihkan dari salinan (`lc-014/garis_hitung.pra-LC014.yml`); cctv01 tidak terdampak (tidak punya `acuan_2` saat LC-008).
- **Perubahan:**
  - `configs/garis_hitung.yml` gerbangpemuda tampilan 0: `[[302,83,463,208]]` — sejajar garis henti, dari kurb median sampai tepi kanan jalur (termasuk lajur sepeda). Dipilih dari 15 kandidat sejajar (`lc-014/kandidat2.py`, gambar lintasan `lc-014/gp_langsung1_jalur.png`): u 0,60 = 11 lintasan vs garis lama 2 pada data yang sama, tinggi kotak median ±54 px. Acuan posisi (siang & malam) dipertahankan. Tampilan 1 tidak diubah.
  - `_entry_langsung.py` `1bc13b7ef5ed1fef`: deteksi `truck` digabung ke kelas `bus` sebelum pelacak (ID stabil); label & skor asli per kotak dipulihkan lewat IoU ≥ 0,3 dengan deteksi frame yang sama.
  - `langsung.py` `5109d9adcb095b1d`: label track = suara terbanyak pengamatan (seri → label yang ada). Bus/truk tetap AMBIGUOUS — tidak ada kelas yang ditebak.
  - `batch.py` `9eb57f969e529646`: `acuan_2` dipertahankan saat garis diganti.
- **Uji:** `tests/test_lintas_besar.py` (3 uji: label berganti → 1 lintasan, suara terbanyak, seri) + 6 pencatat + 11 klasifikasi = 20 lolos di kontainer pada salinan `/tmp`, pencatat DB dimatikan (tidak ada tulisan ke DB).
- **Diverifikasi (malam, 23.18–23.25, 6,3 menit):** garis baru 4 lintasan (2 MP, 1 truk, 1 bus — bus/truk sebelumnya tidak pernah tercatat), garis lama 0 pada data yang sama; server = analisis offline = DB (4 baris). Truk tercatat dengan uid kelompok gabungan (500019) dan label asli `truck`. Posisi kamera cocok 0,958 (acuan malam pulih).
- **Belum diverifikasi:** siang hari (garis dipilih dari data malam); kamera lain belum dianalisis dengan cara yang sama.
- **Konsekuensi:** garis baru di badan jalan hampir tidak dilintasi pejalan kaki — kelas 6 di kamera ini perlu garis pejalan kaki tersendiri (trotoar/zebra), lihat catatan tindak lanjut.
- **Rollback:** salin `lc-014/pra/src/lalin/{_entry_langsung.py,langsung.py,batch.py}` ke server + `docker restart docker-ppvehicle-1`; garis: `PUT /api/garis-hitung/gerbangpemuda` `{"garis":[[35,175,272,359]],"indeks":0}` (acuan kini dipertahankan) atau salin `lc-014/garis_hitung.pra-LC014.yml`.

### LC-015 — Orang di badan jalan → Tidak Dikenal; garis pejalan kaki tersendiri
- **Tanggal:** 26-09-2026 23:36:09 WIB (DB), 23:36:34 WIB (kode + konfigurasi + restart). Atas laporan UAT pemilik proyek ("itu ada sepeda kenapa dibaca pejalan kaki?") dan persetujuan "lanjutkan yang terbaik menurut pilihan kamu".
- **Masalah (diukur):** dalam ±20 menit data malam Gerbang Pemuda, model mendeteksi **0 sepeda**. Pengendara sepeda terdeteksi sebagai `person`; karena sepedanya tidak terdeteksi, penanda pengendara (tumpang dengan kotak sepeda/motor) tidak berlaku dan orang itu tercatat pejalan kaki. Kandidat: "orang" #37 di lajur sepeda hijau (473,145), skor 0,85. Selain itu garis kendaraan LC-014 berada di badan jalan, sehingga pejalan kaki di trotoar tidak pernah dihitung.
- **Perubahan:**
  - `garis_hitung.yml` per tampilan: kunci baru `badan_jalan` (poligon jalur kendaraan + lajur sepeda, TANPA trotoar & zebra) dan `garis_pejalan` (ruas yang hanya menghitung orang). Gerbang Pemuda tampilan 0: `badan_jalan` [[35,175],[480,22],[527,22],[512,70],[495,123],[399,360],[272,360]] (dari marka lajur operator `marka.yml`, kurb median, garis henti); `garis_pejalan` [[545,176,640,143]] di trotoar kanan (dipilih dari 8 kandidat: 17 dari 78 track orang malam ini melintas, terbanyak; `lc-015/uji_orang.py`).
  - `langsung.py`: track `person` yang ≥ 50% pengamatannya di dalam `badan_jalan` → **UNKNOWN** (Tidak Dikenal), `alasan_status = orang_di_badan_jalan` — tidak ditebak sepeda, tidak diklaim pejalan kaki. Ruas `garis_pejalan` hanya untuk orang, nomornya setelah ruas kendaraan; peristiwa membawa `jenis_garis` (kendaraan/pejalan) dan `alasan_status` (juga `yakin_rendah`, `label_detektor_ambigu`).
  - `batch.py`: `load_tampilan_hitung` meneruskan `garis_pejalan`, `badan_jalan`, `oleh`; `simpan_garis_hitung` mempertahankannya saat operator menyimpan garis (tanpa ini keduanya terbuang, cacat sejenis `acuan_2` di LC-014).
  - `bbox.js` (`?v=lc015`): garis pejalan kaki digambar hijau "GARIS PEJALAN KAKI"; panel "Pejalan: a ⇄ b"; log menjelaskan alasan Tidak Dikenal; CSV + kolom `alasan_status`, `jenis_garis`; catatan panel diperbarui.
  - DB (aditif): `ALTER TABLE lalin.lintas_kendaraan ADD COLUMN IF NOT EXISTS alasan_status TEXT, jenis_garis TEXT` (diuji dalam transaksi ROLLBACK dulu; 25 baris lama utuh). DDL `docs/ddl/lintas_kendaraan.sql` + `skema_postgres.sql` = `skema.sql.inc` diperbarui.
- **Uji:** 31 uji lolos di kontainer pada salinan `/tmp` (pencatat DB dimatikan): orang di badan jalan → UNKNOWN; orang di luar badan jalan → pejalan kaki; orang di trotoar → dihitung di garis pejalan (ruas 1); mobil tidak dihitung di garis pejalan; LC-014 tetap; simpan garis mempertahankan `acuan_2`/`garis_pejalan`/`badan_jalan` (berkas sementara).
- **Diverifikasi (Chrome, 23.37–23.41):** garis hijau tampil, panel "Pejalan: 0 ⇄ 0", tanpa galat konsol; orang di badan jalan tampil Tidak Dikenal; lintasan kendaraan tetap tercatat (3 MP + 1 truk, `jenis_garis = kendaraan`). Belum ada pejalan kaki yang melintasi garis hijau selama uji (tengah malam).
- **Batas yang diketahui:** di 3 frame pertama sesi (posisi kamera belum cocok, `aktif = -1`) aturan belum berlaku — tampilan saja, hitungan juga belum berjalan. Orang yang berdiri di kotak kuning gerbang (bagian badan jalan) ikut Tidak Dikenal. `badan_jalan` & `garis_pejalan` baru ada untuk Gerbang Pemuda tampilan 0; kamera lain tetap perilaku lama. Zebra cross belum punya garis pejalan (tidak ada data orang menyeberang untuk memvalidasi posisinya). Akar masalah (sepeda tidak terdeteksi malam) tidak diselesaikan — butuh model/classifier terlatih data DKI.
- **Rollback:** salin `lc-015/pra/src/lalin/{batch.py,langsung.py,simpan_lintas.py}`, `lc-015/pra/src/lalin/web/{dashboard.html,assets/bbox.js}`, `lc-015/pra/configs/garis_hitung.yml` ke server + `docker restart docker-ppvehicle-1`. Kolom DB baru dibiarkan (nullable, tidak mengganggu kode lama).

### LC-016 — Kamera HD dari server Bali Tower kedua (cctv-jsc :8011)
- **Tanggal:** 27-09-2026 00:24:46 WIB (pasang + restart), 00:28 (kanvas & stempel). Atas pilihan pemilik proyek "1&3" dan "ambil yang resolusi baik untuk bahan POC".
- **Latar (diukur):** portal resmi `jakcctv.jakarta.go.id/publik` memuat 50 kamera: 46 di `dki-jkt…:7028` (640×360) dan 4 di `cctv-jsc.balitower.co.id:8011` (720p–1080p). 10 tautan Bali Tower lain yang dipublikasikan detik.com (28-08-2025) hidup di host yang sama (hingga 3200×1800). CORS `*`, satu varian per kamera. Tidak ada penebakan ID. Daftar 60 kamera terverifikasi: `docs/dishub/daftar-cctv-publik-jakarta-2026-09-27.xlsx`.
- **Uji mesin Live yang sama** (malam 00:13–00:25, ±45 dtk/kamera, bergiliran; `lc-016/uji_kamera.sh`, `analisis_kamera.py`): % deteksi di atas ambang — Senayan 001 69%, Bendungan Hilir 4 65%, Bendungan Hilir 2 61% (53 truk, 13 bus), Kebon Melati 45%, Senayan 018 29%, Kuningan Barat 22%; pembanding Gerbang Pemuda 17%, Jati Baru 0 deteksi. Beban dekode HD 0,11–0,54 core.
- **Perubahan:** `cameras.yml` +6 kamera HD (`base_url` per kamera); `batch.py` `BASE_URL_SAH` — alamat server per kamera hanya dari `cameras.yml` dan hanya dua host Bali Tower (bukan dari suntingan UI: server menyambung ke alamat ini); `api.py` cache snapshot 60 dtk per kamera (gambar mini kamera tanpa `preview.jpg`); `dashboard.html` memutar `stream` per kamera, gambar mini lewat server, stempel menampilkan resolusi video sebenarnya; `bbox.js` (`?v=lc016`) kanvas selalu 640×360 — sebelumnya = ukuran video, pada 3200×1800 garis 1,7 px & huruf 9 px diperkecil ke ±0,4 px sehingga kotak tidak terlihat.
- **Uji:** 35 uji lolos (termasuk 4 uji `test_kamera_base_url.py`: host tak terdaftar & suntingan UI ditolak). Chrome: video 3200×1800 berputar, kotak & label tampil, tanpa galat konsol.
- **Temuan terbuka:** pada kamera HD detektor ±1 dtk/frame (> interval 0,6 dtk) → cakupan ±82% waktu; butuh GPU untuk "tanpa terlewat" (lihat LC-017). Kamera HD belum punya garis hitung. Cikoko (4:3) tidak dimasukkan (pemutar 16:9).
- **Rollback:** salin `lc-016/pra/…` (batch.py, api.py, dashboard.html, cameras.yml; bbox.js = `lc-016/pra/bbox.js.pra`) + restart.

### LC-017 — Cegah proses deteksi yatim + batasi thread dekoder
- **Tanggal:** 27-09-2026 00:41:37 WIB (restart).
- **Masalah (diukur):** dua permintaan "mulai Live" hampir bersamaan membuat dua proses `_entry_langsung.py`; pengelola hanya melacak yang terakhir. Ditemukan 2 proses Bendungan Hilir 2 yatim (±1,2 GB dan ±60% CPU masing-masing, termasuk ffmpeg-nya); kontainer 4,7/5 GB memori. Keduanya dihentikan (hanya proses milik kita, dikenali dari `LALIN_SOURCE`); memori turun ke 2,3 GB.
- **Perubahan:** `langsung.py` `c648e6f3ee6e2221`: `mulai()` dibungkus `_kunci_mulai`. `_entry_langsung.py` `eeae15ffe0497ce9`: ffmpeg `-threads 2` — uji Bendungan Hilir 2 (±60 dtk/varian): tunda median 7,7 → 1,0 dtk, celah > 1,6 dtk 5 → 1; cakupan tetap ±82% (batas CPU detektor).
- **Rollback:** salin `lc-017/pra/src/lalin/{langsung.py,_entry_langsung.py}` + restart.

### LC-018 — Uji recall deteksi untuk satu kamera HD Benhil2
- **Tanggal:** 28-09-2026 (uji POC satu kamera).
- **Pemicu:** API server menunjukkan Live sebelumnya aktif di `gerbangpemuda` saat POC diarahkan ke `benhil2`. Sesudah dialihkan, garis cocok (skor posisi > 0,95), DB sehat, tetapi 13 lintasan awal hanya berlabel `bus`/`car`; belum ada lintasan `motorcycle`.
- **Perubahan:** ambang kandidat detektor Live diturunkan dari 0,30 ke 0,20 untuk uji recall di pipeline satu kamera. Ambang klasifikasi motorcycle 0,45 dan aturan porsi yakin 50% tidak berubah; kandidat lemah tetap UNKNOWN. Tracker sudah memakai `conf_thres: 0.20`.
- **Batas uji:** berlaku pada pipeline Live tunggal (saat ini Benhil2); hanya nilai ambang deteksi yang berubah. Ukur jeda, lintasan, dan status label; rollback bila derau bertambah atau laju turun.
- **Sebelum pasang:** `langsung.py` SHA-256 `c648e6f3ee6e2221`; salinan di `lc-018/pra/src/lalin/` dan server `langsung.py.lc018-pra-20260928`.
- **Dipasang & diperiksa 28-09 10:04–10:08 WIB:** API menunjukkan `benhil2`, tampilan 0 aktif, skor cocok 0,939, jeda 0,57 dtk, tiada galat DB; setelah 45 frame ada 2 lintasan mobil. Sesudahnya 20 lintasan tercatat dalam buffer (12 mobil terklasifikasi, 3 mobil UNKNOWN, 2 bus AMBIGUOUS, 3 bus UNKNOWN); belum ada label detektor `motorcycle`. CPU kontainer 634% dari batas 800%, memori 1,48/5 GB, antrean DB 0. Ini masih pengamatan singkat, belum bukti peningkatan akurasi.
- **Dataset:** restart kontainer mematikan proses pengumpul; proses dijalankan lagi untuk 2,72 jam tersisa. Berkas lama tetap di server, berkas baru terlihat pada 10:05 WIB.
- **Rollback:** ubah `env["LALIN_THRESHOLD"]` kembali ke `"0.30"`, lalu restart kontainer `docker-ppvehicle-1` dan pastikan collector dataset dilanjutkan.

### LC-019 — Jejak metadata lintasan untuk pelabelan Benhil2
- **Tanggal:** 28-09-2026 13:10 WIB (DDL, kode, dan restart `docker-ppvehicle-1`).
- **Tujuan:** setiap label manusia dapat dipasangkan ke kendaraan yang tepat melalui `(sesi_id, seq, track_id, PTS)`, tanpa mengandalkan IoU dengan keyframe HD yang diambil beberapa detik berbeda.
- **Perubahan DB (aditif):** kolom nullable `jejak JSONB` pada `lalin.lintas_kendaraan`. Isinya daftar pendek pengamatan track: `pts`, `bbox` 640×360, `skor`, dan `label` detektor. Tidak ada gambar, pelat, wajah, maupun identitas.
- **Perubahan Live:** `langsung.py` menyimpan maksimal 80 pengamatan (±48 detik pada interval 0,6 detik) untuk setiap track. Saat kendaraan melewati garis dan peristiwa dilepas, salinan jejak dimasukkan ke baris lintasan melalui pencatat DB yang sudah ada. Mesin deteksi, tracker, garis hitung, ambang, dan keputusan kelas tidak diubah.
- **Verifikasi:** sesi `1790575910701`, seq 1–2, masing-masing berisi 7 dan 6 titik jejak; PTS awal/akhir, bbox, skor, dan label `car` terbaca dari PostgreSQL. Antrean DB nol dan tidak ada galat.
- **Antrean label:** skrip `scripts/siapkan_antrean_label.py` membentuk berkas metadata-only di `/app/runs/dataset/labeling/benhil2-antrean-label.json`, seimbang per label detektor (`motorcycle`, `car`, `bus`, `truck`, `person`, `bicycle`). Pada pembuatan pertama, dua `car` tersedia; kelas lain akan masuk ketika lintasan berjejak terkumpul. Berkas izin 600, folder izin 700.
- **Rollback:** salin versi pra-LC-019 untuk `src/lalin/langsung.py` dan `src/lalin/simpan_lintas.py`, lalu restart `docker-ppvehicle-1`. Kolom `jejak` dibiarkan nullable; kode lama tetap dapat membaca dan menulis baris tanpa nilai itu.

### DS-001 — Pengumpulan dataset mentah (bukan perubahan perilaku Live)
- **Mulai:** 27-09-2026 00:42 WIB (6 kamera), **00:43:42 WIB hanya Bendungan Hilir 2** (keputusan pemilik proyek: satu kamera HD untuk POC). Berjalan 36 jam atau sampai sisa disk < 100 GB.
- **Cara:** `src/lalin/pengumpul_dataset.py` di kontainer (`docker exec -d … python -m lalin.pengumpul_dataset --jam 36 --kamera benhil2`), ffmpeg hanya keyframe (`-skip_frame nokey`, 1 thread, nice 19), frame resolusi asli ±1,2 MB tiap ±5–8 dtk → `/app/runs/dataset/mentah/benhil2/<WIB>.jpg`. Log: `/app/runs/dataset/pengumpul.log`.
- **Data pribadi:** frame 3200×1800 dapat memuat pelat & wajah → folder izin 700 (root kontainer), tidak disalin ke laptop, tidak dipublikasikan; pelabelan memakai potongan kendaraan yang diperkecil.
- **Catatan:** proses berhenti bila kontainer di-restart — jalankan ulang perintah di atas. Frame 5 kamera HD lain (±38 MB, 00:37–00:43) tidak dipakai; tidak dihapus (penghapusan perlu persetujuan).

## Hal yang belum memenuhi standar register
- LC-001 s.d. LC-005a tidak punya salinan pra-perubahan penuh karena tidak ada kontrol versi. Mulai LC-006, setiap perubahan disiapkan sebagai salinan berkas + manifest SHA-256 sebelum dipasang (atau commit git bila disetujui). LC-006 sudah mengikuti aturan ini.
- Ambang LC-005 masih ditanam di kode; langkah 4 (lapisan pemetaan pusat berbasis konfigurasi) akan memindahkannya ke satu berkas konfigurasi berversi.
