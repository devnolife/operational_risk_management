# 🚀 Prompt Deploy ke Server VPS

Salin **seluruh isi blok prompt di bawah** lalu tempel ke AI assistant di server VPS
Anda (GitHub Copilot CLI, Claude, ChatGPT, dsb.) — atau ikuti *cheat sheet* manual
di bagian bawah file ini.

---

## 📋 Prompt (salin dari sini)

```text
Deploy-kan project Streamlit berikut di VPS Ubuntu/Debian ini sampai bisa diakses
publik. Kerjakan langkah demi langkah, verifikasi tiap langkah sebelum lanjut.

KONTEKS PROJECT
- Repo GitHub : https://github.com/devnolife/operational_risk_management
- Aplikasi    : dashboard Streamlit (app.py) — Operational Risk Management,
                implementasi Giudici (2009) Bab 12, project kuliah Data Mining S2.
- Python      : 3.10 atau lebih baru.
- Dependensi  : requirements.txt (pandas, numpy, streamlit, plotly, scikit-learn, dll).
- PENTING     : folder data/ di-gitignore. Aplikasi WAJIB punya file
                data/processed/paysim_processed.parquet sebelum dijalankan.
                File itu dibuat dengan menjalankan dua perintah (lihat langkah 4).

LANGKAH YANG HARUS DIKERJAKAN

1. Persiapan sistem:
   - sudo apt update && sudo apt install -y python3 python3-venv python3-pip git nginx
   - Pastikan python3 --version >= 3.10.

2. Clone project:
   - git clone https://github.com/devnolife/operational_risk_management.git /opt/orm
   - cd /opt/orm

3. Virtual environment & dependensi:
   - python3 -m venv .venv
   - source .venv/bin/activate
   - pip install --upgrade pip
   - pip install -r requirements.txt

4. Siapkan data (WAJIB, pilih salah satu):
   a. Data contoh (cepat, tanpa akun Kaggle — pakai ini secara default):
      - python -m src.make_sample_data
      - python -m src.data
   b. Dataset PaySim asli dari Kaggle (opsional, butuh kaggle.json):
      - letakkan kaggle.json di ~/.kaggle/ lalu:
      - kaggle datasets download -d ealaxi/paysim1 -p data/raw --unzip
      - python -m src.data
   Verifikasi: file data/processed/paysim_processed.parquet harus ada.

5. Tes aplikasi berjalan:
   - streamlit run app.py --server.headless true --server.port 8501
   - curl -s -o /dev/null -w "%{http_code}" http://localhost:8501  → harus 200
   - hentikan dulu (Ctrl+C) sebelum lanjut ke systemd.

6. Buat systemd service agar otomatis hidup setelah reboot:
   - File /etc/systemd/system/orm-dashboard.service berisi:
       [Unit]
       Description=Operational Risk Management Streamlit Dashboard
       After=network.target
       [Service]
       Type=simple
       User=www-data
       WorkingDirectory=/opt/orm
       ExecStart=/opt/orm/.venv/bin/streamlit run app.py --server.headless true --server.port 8501 --server.address 127.0.0.1
       Restart=always
       RestartSec=5
       [Install]
       WantedBy=multi-user.target
   - chown -R www-data:www-data /opt/orm
   - sudo systemctl daemon-reload && sudo systemctl enable --now orm-dashboard
   - Verifikasi: systemctl status orm-dashboard → active (running).

7. Nginx reverse proxy (Streamlit BUTUH WebSocket — header upgrade wajib):
   - File /etc/nginx/sites-available/orm berisi:
       server {
           listen 80;
           server_name _;   # ganti dengan domain bila ada
           location / {
               proxy_pass http://127.0.0.1:8501;
               proxy_http_version 1.1;
               proxy_set_header Upgrade $http_upgrade;
               proxy_set_header Connection "upgrade";
               proxy_set_header Host $host;
               proxy_set_header X-Real-IP $remote_addr;
               proxy_read_timeout 86400;
           }
       }
   - sudo ln -s /etc/nginx/sites-available/orm /etc/nginx/sites-enabled/
   - sudo rm -f /etc/nginx/sites-enabled/default
   - sudo nginx -t && sudo systemctl reload nginx

8. Firewall:
   - sudo ufw allow OpenSSH && sudo ufw allow 'Nginx Full' && sudo ufw enable

9. (Opsional, bila ada domain) HTTPS dengan certbot:
   - sudo apt install -y certbot python3-certbot-nginx
   - sudo certbot --nginx -d NAMA_DOMAIN

KRITERIA SELESAI (verifikasi semuanya):
- curl http://IP_SERVER mengembalikan HTTP 200.
- Dashboard terbuka di browser; sidebar memuat 8 halaman:
  Beranda, Data Lengkap, 1—Scorecard, 2—VaR Aktuaria, 3—VaR Integrasi & BIA,
  Ringkasan Eksekutif, Jalankan Program, Buku Referensi.
- Halaman "Jalankan Program": tombol 🚀 berhasil mengeksekusi modul orm.scorecard.
- sudo reboot lalu cek dashboard hidup kembali otomatis.

Jika ada error, tampilkan log-nya (journalctl -u orm-dashboard -n 50) lalu perbaiki.
```

---

## ⚡ Cheat Sheet Manual (tanpa AI)

```bash
# 1. Sistem & clone
sudo apt update && sudo apt install -y python3 python3-venv python3-pip git nginx
sudo git clone https://github.com/devnolife/operational_risk_management.git /opt/orm
cd /opt/orm

# 2. Environment & dependensi
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt

# 3. Data (wajib sebelum app jalan)
python -m src.make_sample_data && python -m src.data

# 4. Tes cepat
streamlit run app.py --server.headless true --server.port 8501
# buka http://IP_SERVER:8501 (buka port 8501 dulu: sudo ufw allow 8501)
```

Untuk produksi (auto-start + port 80), ikuti langkah 6–8 pada prompt di atas
(systemd service + nginx reverse proxy + ufw).

## 📝 Catatan

| Hal | Keterangan |
|---|---|
| Spesifikasi VPS minimal | 1 vCPU, 2 GB RAM (simulasi Monte Carlo 20k skenario) |
| Data contoh vs asli | `src.make_sample_data` cukup untuk demo; PaySim asli (~6 juta baris, unduh dari Kaggle) butuh RAM lebih besar saat preprocessing |
| Port internal | Streamlit jalan di `127.0.0.1:8501`, diekspos lewat nginx port 80/443 |
| Log aplikasi | `journalctl -u orm-dashboard -f` |
| Update versi baru | `cd /opt/orm && git pull && sudo systemctl restart orm-dashboard` |
