#!/bin/sh
# Dijalankan SEKALI oleh image postgres saat volume datanya masih kosong.
# Kata sandi datang dari docker/.env.pg (dibuat acak di server, izin 600) -
# tidak pernah ditulis di repositori.
set -eu
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
     -v app_pw="$LALIN_APP_PASSWORD" -v baca_pw="$LALIN_BACA_PASSWORD" <<'SQL'
-- Aplikasi: pemilik skema, boleh menulis.
CREATE ROLE lalin_app  LOGIN PASSWORD :'app_pw';
-- Tim DB: hanya baca. Tidak bisa mengubah kejadian, keputusan verifikasi,
-- atau aduan - keduanya bukti yang harus tetap utuh.
CREATE ROLE lalin_baca LOGIN PASSWORD :'baca_pw';

REVOKE CREATE ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON DATABASE lalin FROM PUBLIC;
GRANT CONNECT ON DATABASE lalin TO lalin_app, lalin_baca;
-- CREATE SCHEMA IF NOT EXISTS di skema.sql tetap menuntut hak CREATE pada
-- basis data, meskipun skemanya sudah ada.
GRANT CREATE ON DATABASE lalin TO lalin_app;

CREATE SCHEMA lalin AUTHORIZATION lalin_app;
GRANT USAGE ON SCHEMA lalin TO lalin_baca;
ALTER DEFAULT PRIVILEGES FOR ROLE lalin_app IN SCHEMA lalin
      GRANT SELECT ON TABLES TO lalin_baca;
ALTER ROLE lalin_app  SET search_path = lalin;
ALTER ROLE lalin_baca SET search_path = lalin;
-- Pengaman kedua bagi akun baca: setiap transaksinya read-only.
ALTER ROLE lalin_baca SET default_transaction_read_only = on;
SQL

# Skema dibuat SEBAGAI lalin_app supaya seluruh tabel miliknya, dan hak
# SELECT bawaan di atas ikut berlaku untuk lalin_baca.
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
     -c "SET ROLE lalin_app;" -f /docker-entrypoint-initdb.d/skema.sql.inc
