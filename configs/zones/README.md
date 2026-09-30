# Zona (region polygon)

`region_polygon` adalah daftar koordinat piksel searah jarum jam pada resolusi
**asli video**, bukan resolusi tampilan: `x0 y0 x1 y1 x2 y2 ...` (minimal 3 titik).

Dipakai oleh skenario `break_in` dan `illegal_parking`, dan oleh flag
`--region-type custom`.

## Cara bikin koordinatnya

```bash
python scripts/pick_zone.py data/samples/jalan.mp4
```

Klik titik-titik zona searah jarum jam di jendela yang muncul, tekan `s` untuk
simpan, `u` untuk undo titik terakhir, `q` untuk keluar. Script mencetak string
yang bisa langsung ditempel ke config atau ke CLI.

## Contoh

`example.json` berisi zona persegi untuk video 1920x1080. Muat isinya ke config
skenario atau lempar lewat CLI:

```bash
python -m ppvehicle run --scenario illegal_parking --source data/samples/jalan.mp4 --region-polygon 600 300 1300 300 1300 800 600 800 --illegal-parking-time 10
```
