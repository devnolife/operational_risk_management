"""
generate_reports.py — Pembuat laporan PDF per model pengukuran risiko operasional.

Menghasilkan TIGA file PDF terpisah (satu per model / penanggung jawab), mengikuti
metodologi Giudici (2009) Bab 12. Setiap laporan MENJALANKAN modul `orm/` pada data
PaySim yang sudah diproses, lalu menyajikan penjelasan rinci + angka nyata + grafik:

  1. Laporan_Model1_Scorecard_LisIndriani.pdf            (orm/scorecard.py, Gbr 12.1)
  2. Laporan_Model2_Aktuaria_AnaSulistianaAlwi.pdf       (orm/actuarial.py, Gbr 12.2)
  3. Laporan_Model3_IntegrasiBIA_AndiAgungDwiAryaB.pdf   (orm/integrated.py, Gbr 12.2)

Tiap laporan memuat: latar belakang, landasan teori + rumus, dataset, alur proses,
CONTOH PERHITUNGAN langkah-demi-langkah dengan angka nyata, hasil + grafik,
glosarium istilah, interpretasi, serta keterbatasan & saran.

Jalankan:
    python generate_reports.py
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from io import BytesIO
from pathlib import Path
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# --- Modul model (orm) ---
from orm.actuarial import actuarial_summary
from orm.categories import SCALES, rank_to_letter
from orm.data_sources import (
    generate_expert_opinions,
    generate_external_losses,
    internal_frequency_severity,
    load_internal_losses,
    scale_external_to_internal,
)
from orm.integrated import run_comparison
from orm.scorecard import (
    build_scorecard,
    consensus_multiplicity,
    median_class_rank,
    normalized_gini,
    self_assessment_total_loss,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MARGIN = 2 * cm
CONTENT_WIDTH = A4[0] - 2 * MARGIN
INK = colors.HexColor("#222222")
MUTED = colors.HexColor("#5b6470")

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.edgecolor": "#d0d4da",
    "axes.grid": True,
    "grid.color": "#e7eaee",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


# ---------------------------------------------------------------------------
# Util
# ---------------------------------------------------------------------------
def rp(x: float) -> str:
    """Format mata uang gaya dashboard (Rp + pemisah ribuan)."""
    return f"Rp {x:,.0f}"


def mono(text: str) -> str:
    """Bungkus teks sebagai monospace (untuk rumus inline)."""
    return f'<font face="Courier">{text}</font>'


def tint(hex_color: str, factor: float) -> colors.Color:
    """Campur warna menuju putih (factor 0..1, makin besar makin terang)."""
    c = colors.HexColor(hex_color)
    return colors.Color(
        c.red + (1 - c.red) * factor,
        c.green + (1 - c.green) * factor,
        c.blue + (1 - c.blue) * factor,
    )


def mpl_image(fig, width: float = CONTENT_WIDTH) -> Image:
    """Render figure matplotlib menjadi flowable Image (skala menjaga rasio)."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    iw, ih = ImageReader(buf).getSize()
    buf.seek(0)
    return Image(buf, width=width, height=width * ih / iw)


# ---------------------------------------------------------------------------
# Gaya paragraf
# ---------------------------------------------------------------------------
_BASE = getSampleStyleSheet()


def styles(accent: str) -> dict:
    a = colors.HexColor(accent)
    return {
        "band_kicker": ParagraphStyle(
            "kicker", parent=_BASE["Normal"], fontName="Helvetica-Bold",
            fontSize=8.5, textColor=colors.white, leading=11, spaceAfter=2),
        "band_title": ParagraphStyle(
            "btitle", parent=_BASE["Normal"], fontName="Helvetica-Bold",
            fontSize=19, textColor=colors.white, leading=22),
        "band_sub": ParagraphStyle(
            "bsub", parent=_BASE["Normal"], fontName="Helvetica",
            fontSize=10.5, textColor=colors.white, leading=14, spaceBefore=4),
        "h": ParagraphStyle(
            "h", parent=_BASE["Normal"], fontName="Helvetica-Bold",
            fontSize=13, textColor=a, leading=16, spaceBefore=4, spaceAfter=4),
        "h2": ParagraphStyle(
            "h2", parent=_BASE["Normal"], fontName="Helvetica-Bold",
            fontSize=10.5, textColor=INK, leading=14, spaceBefore=6, spaceAfter=2),
        "body": ParagraphStyle(
            "body", parent=_BASE["Normal"], fontName="Helvetica",
            fontSize=10, textColor=INK, leading=15, alignment=TA_JUSTIFY,
            spaceAfter=6),
        "bullet": ParagraphStyle(
            "bullet", parent=_BASE["Normal"], fontName="Helvetica",
            fontSize=10, textColor=INK, leading=15, leftIndent=14,
            bulletIndent=2, spaceAfter=3, alignment=TA_JUSTIFY),
        "caption": ParagraphStyle(
            "caption", parent=_BASE["Normal"], fontName="Helvetica-Oblique",
            fontSize=8.5, textColor=MUTED, leading=12, alignment=TA_CENTER,
            spaceBefore=3, spaceAfter=8),
        "metric_val": ParagraphStyle(
            "mval", parent=_BASE["Normal"], fontName="Helvetica-Bold",
            fontSize=14, textColor=a, leading=17, alignment=TA_CENTER),
        "metric_lab": ParagraphStyle(
            "mlab", parent=_BASE["Normal"], fontName="Helvetica",
            fontSize=8, textColor=MUTED, leading=10, alignment=TA_CENTER),
        "cell": ParagraphStyle(
            "cell", parent=_BASE["Normal"], fontName="Helvetica",
            fontSize=8.5, textColor=INK, leading=11),
        "cell_b": ParagraphStyle(
            "cellb", parent=_BASE["Normal"], fontName="Helvetica-Bold",
            fontSize=8.5, textColor=colors.white, leading=11),
        "note_title": ParagraphStyle(
            "ntitle", parent=_BASE["Normal"], fontName="Helvetica-Bold",
            fontSize=10.5, textColor=a, leading=14, spaceAfter=3),
        "note": ParagraphStyle(
            "note", parent=_BASE["Normal"], fontName="Helvetica",
            fontSize=9.3, textColor=INK, leading=14, alignment=TA_LEFT,
            spaceAfter=3),
        "step": ParagraphStyle(
            "step", parent=_BASE["Normal"], fontName="Helvetica",
            fontSize=9.6, textColor=INK, leading=14, leftIndent=16,
            spaceAfter=4, alignment=TA_JUSTIFY),
    }


# ---------------------------------------------------------------------------
# Komponen flowable
# ---------------------------------------------------------------------------
def title_band(accent, st, kicker, title, members_line, meta_pairs):
    a = colors.HexColor(accent)
    inner = [
        [Paragraph(kicker, st["band_kicker"])],
        [Paragraph(title, st["band_title"])],
        [Paragraph(members_line, st["band_sub"])],
    ]
    band = Table(inner, colWidths=[CONTENT_WIDTH])
    band.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), a),
        ("LEFTPADDING", (0, 0), (-1, -1), 16),
        ("RIGHTPADDING", (0, 0), (-1, -1), 16),
        ("TOPPADDING", (0, 0), (0, 0), 14),
        ("BOTTOMPADDING", (-1, -1), (-1, -1), 14),
        ("TOPPADDING", (0, 1), (-1, -1), 2),
    ]))
    meta_cells = []
    for label, value in meta_pairs:
        meta_cells.append(Paragraph(
            f'<font size=7 color="#8a929c">{label}</font><br/>'
            f'<font size=9 color="#222222"><b>{value}</b></font>',
            st["cell"]))
    meta = Table([meta_cells], colWidths=[CONTENT_WIDTH / len(meta_cells)] * len(meta_cells))
    meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), tint(accent, 0.90)),
        ("LINEBELOW", (0, 0), (-1, -1), 2, a),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return [band, meta, Spacer(1, 12)]


def heading(text, st):
    a = st["h"].textColor
    return KeepTogether([
        Paragraph(text, st["h"]),
        HRFlowable(width="100%", thickness=1.1, color=a, spaceBefore=1, spaceAfter=8),
    ])


def bullets(items, st):
    return [Paragraph(f"&bull;&nbsp;&nbsp;{t}", st["bullet"]) for t in items]


def note_box(accent, st, title, body_flowables):
    """Kotak callout (latar tint + garis aksen kiri) untuk contoh/perhitungan."""
    a = colors.HexColor(accent)
    inner = [[Paragraph(title, st["note_title"])]]
    for fl in body_flowables:
        inner.append([fl])
    t = Table(inner, colWidths=[CONTENT_WIDTH])
    style = [
        ("BACKGROUND", (0, 0), (-1, -1), tint(accent, 0.93)),
        ("LINEBEFORE", (0, 0), (0, -1), 3.2, a),
        ("LEFTPADDING", (0, 0), (-1, -1), 13),
        ("RIGHTPADDING", (0, 0), (-1, -1), 13),
        ("TOPPADDING", (0, 0), (0, 0), 9),
        ("TOPPADDING", (0, 1), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -2), 1),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 9),
    ]
    t.setStyle(TableStyle(style))
    return [t, Spacer(1, 8)]


def metric_cards(accent, st, cards):
    """cards: list of (value, label)."""
    row = []
    for value, label in cards:
        row.append([Paragraph(value, st["metric_val"]),
                    Paragraph(label, st["metric_lab"])])
    inner = [[Table([[v], [l]], style=TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
    ])) for v, l in row]]
    n = len(cards)
    t = Table(inner, colWidths=[CONTENT_WIDTH / n] * n)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), tint(accent, 0.88)),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(accent)),
        ("INNERGRID", (0, 0), (-1, -1), 0.8, colors.white),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def data_table(accent, st, header, rows, col_widths, aligns=None):
    a = colors.HexColor(accent)
    head = [Paragraph(h, st["cell_b"]) for h in header]
    body = [[Paragraph(str(c), st["cell"]) for c in r] for r in rows]
    t = Table([head] + body, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), a),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#dfe3e8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    for i in range(1, len(body) + 1):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), tint(accent, 0.93)))
    for col, al in (aligns or {}).items():
        style.append(("ALIGN", (col, 1), (col, -1), al))
    t.setStyle(TableStyle(style))
    return t


def glossary(accent, st, pairs):
    rows = [[f"<b>{term}</b>", desc] for term, desc in pairs]
    return data_table(accent, st, ["Istilah", "Penjelasan singkat"], rows,
                      [4.3 * cm, 11.9 * cm])


def page_decorator(accent, model_label, member):
    a = colors.HexColor(accent)

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(a)
        canvas.setLineWidth(0.8)
        canvas.line(MARGIN, 1.5 * cm, A4[0] - MARGIN, 1.5 * cm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, 1.05 * cm,
                          "Operational Risk Management - Data Mining S2  |  Giudici (2009) Bab 12")
        canvas.drawRightString(A4[0] - MARGIN, 1.05 * cm,
                               f"{model_label} - {member}  |  Hal. {doc.page}")
        canvas.restoreState()

    return decorate


# ---------------------------------------------------------------------------
# Diagram alur (input -> proses -> output)
# ---------------------------------------------------------------------------
def flow_diagram(steps, accent_hex):
    n = len(steps)
    fig, ax = plt.subplots(figsize=(7.8, 1.75))
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")
    a = accent_hex
    fills = [tint(a, 0.86), tint(a, 0.70), tint(a, 0.40), tint(a, 0.18)]
    bw = 0.86
    for i, (title, sub) in enumerate(steps):
        x = i + (1 - bw) / 2
        fill = fills[min(i, len(fills) - 1)]
        text_col = "white" if i >= 2 else "#1c2733"
        box = FancyBboxPatch(
            (x, 0.22), bw, 0.56,
            boxstyle="round,pad=0.012,rounding_size=0.04",
            linewidth=1.3, edgecolor=a,
            facecolor=(fill.red, fill.green, fill.blue))
        ax.add_patch(box)
        ax.text(i + 0.5, 0.60, title, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=text_col)
        ax.text(i + 0.5, 0.40, sub, ha="center", va="center",
                fontsize=7.6, color=text_col)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch(
                (x + bw + 0.005, 0.5), (i + 1 + (1 - bw) / 2 - 0.005, 0.5),
                arrowstyle="-|>", mutation_scale=14, linewidth=1.6, color=a))
    fig.tight_layout(pad=0.2)
    return mpl_image(fig)


# ---------------------------------------------------------------------------
# Build dokumen
# ---------------------------------------------------------------------------
def build_pdf(filename, accent, model_label, member, story_factory):
    path = PROJECT_ROOT / filename
    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=1.6 * cm, bottomMargin=2.0 * cm,
        title=f"Laporan {model_label} - {member}",
        author=member,
        subject="Operational Risk Management - Giudici (2009) Bab 12",
    )
    deco = page_decorator(accent, model_label, member)
    doc.build(story_factory(), onFirstPage=deco, onLaterPages=deco)
    return path


# ===========================================================================
# MODEL 1 — SCORECARD / SELF-ASSESSMENT  (Lis Indriani)
# ===========================================================================
def report_scorecard():
    accent = "#1f9d55"  # hijau (tema lampu lalu lintas)
    st = styles(accent)
    member = "Lis Indriani"

    # --- Jalankan model ---
    opinions = generate_expert_opinions(n_experts=8, seed=42)
    sc = build_scorecard(opinions)
    total_loss = self_assessment_total_loss(sc)
    n_experts = int(opinions["expert"].nunique())
    n_cat = int(opinions.groupby(["business_line", "event_type"]).ngroups)
    red_sev = int((sc["severity_color"] == "red").sum())
    top = sc.head(10)

    # --- Contoh perhitungan: dimensi severity pada kategori prioritas teratas ---
    wb, we = top.iloc[0]["business_line"], top.iloc[0]["event_type"]
    sev_votes = opinions[(opinions.business_line == wb)
                         & (opinions.event_type == we)]["severity"].tolist()
    sev_classes = SCALES["severity"]
    K = len(sev_classes)
    sev_rank = median_class_rank(sev_votes, "severity")
    sev_median_class = sev_classes[sev_rank - 1]
    sev_letter = rank_to_letter(sev_rank, K)
    sev_gini = normalized_gini(sev_votes, "severity")
    sev_mult = consensus_multiplicity(sev_gini)
    sev_rating = sev_letter * sev_mult
    cnt = Counter(sev_votes)
    counts_ordered = [(c, cnt[c]) for c in sev_classes if cnt.get(c, 0)]
    ranks_sorted = sorted(sev_classes.index(v) + 1 for v in sev_votes)
    p_sq_sum = sum((v / len(sev_votes)) ** 2 for _, v in counts_ordered)
    gini_raw = 1 - p_sq_sum
    psq_str = " + ".join(f"({v}/{len(sev_votes)})^2" for _, v in counts_ordered)
    counts_str = ", ".join(f"{c}={v}" for c, v in counts_ordered)
    median_pos = len(sev_votes) // 2 + 1
    rule_txt = {
        3: "tidak lebih dari 1/3 (konsensus tinggi), sehingga huruf dilipat 3 kali",
        2: "berada antara 1/3 dan 2/3 (konsensus sedang), sehingga huruf dilipat 2 kali",
        1: "lebih dari 2/3 (konsensus rendah), sehingga huruf tetap tunggal",
    }[sev_mult]

    # --- Chart 1: Top-10 perceived loss ---
    fig1, ax1 = plt.subplots(figsize=(7.6, 3.6))
    labels = [f"{r.event_type[:26]}\n({r.business_line[:18]})" for r in top.itertuples()]
    vals = top["perceived_loss"].to_numpy() / 1e6
    ax1.barh(range(len(top)), vals, color="#1f9d55", edgecolor="#13713c")
    ax1.set_yticks(range(len(top)))
    ax1.set_yticklabels(labels, fontsize=7.2)
    ax1.invert_yaxis()
    ax1.set_xlabel("Perkiraan kerugian (juta Rp)")
    ax1.set_title("10 Kategori Prioritas Intervensi Tertinggi (perceived loss)")
    for i, v in enumerate(vals):
        ax1.text(v, i, f" {v:,.0f}", va="center", fontsize=7)
    chart1 = mpl_image(fig1)

    # --- Chart 2: distribusi lampu lalu lintas per dimensi ---
    dims = [("Frekuensi", "frequency_color"), ("Severity", "severity_color"),
            ("Kontrol", "control_color")]
    cmap = {"green": "#2ca02c", "yellow": "#f1c40f", "red": "#d62728"}
    nice = {"green": "A (hijau)", "yellow": "B (kuning)", "red": "C+ (merah)"}
    fig2, ax2 = plt.subplots(figsize=(7.6, 3.0))
    xpos = np.arange(len(dims))
    w = 0.26
    for j, co in enumerate(["green", "yellow", "red"]):
        counts = [int((sc[col] == co).sum()) for _, col in dims]
        ax2.bar(xpos + (j - 1) * w, counts, w, label=nice[co],
                color=cmap[co], edgecolor="#555")
        for k, c in enumerate(counts):
            ax2.text(xpos[k] + (j - 1) * w, c, str(c), ha="center",
                     va="bottom", fontsize=7)
    ax2.set_xticks(xpos)
    ax2.set_xticklabels([d[0] for d in dims])
    ax2.set_ylabel("Jumlah kategori")
    ax2.set_title("Sebaran Rating Lampu Lalu Lintas (56 kategori Basel)")
    ax2.legend(fontsize=8, ncol=3, loc="upper center", frameon=False)
    chart2 = mpl_image(fig2)

    flow = flow_diagram([
        ("INPUT", "Opini 8 expert\n56 kategori Basel"),
        ("MEDIAN KELAS", "tetapkan huruf\nA / B / C ..."),
        ("INDEKS GINI", "konsensus ->\nAAA / AA / A"),
        ("OUTPUT", "Rating + lampu\nlalu lintas"),
    ], accent)

    top_rows = []
    for i, r in enumerate(top.itertuples(), 1):
        top_rows.append([
            i, r.business_line, r.event_type,
            r.frequency_rating, r.severity_rating, r.control_rating,
            f"{r.perceived_loss/1e6:,.1f}", int(r.priority_score),
        ])

    def story():
        s = []
        s += title_band(
            accent, st,
            "LAPORAN MODEL 1 - PENGUKURAN RISIKO OPERASIONAL",
            "Model Scorecard / Self-Assessment",
            "Penanggung jawab: <b>Lis Indriani</b>",
            [("Modul kode", "orm/scorecard.py"),
             ("Acuan buku", "Giudici 2009, Gbr 12.1"),
             ("Tanggal", datetime.now().strftime("%d %b %Y"))])

        s += [heading("1. Latar Belakang &amp; Rumusan Masalah", st)]
        s += [Paragraph(
            "Risiko operasional adalah risiko kerugian akibat ketidakcukupan atau "
            "kegagalan proses internal, manusia, sistem, atau kejadian eksternal. "
            "Tidak seperti risiko pasar atau kredit, banyak kategori risiko operasional "
            "<b>jarang terjadi namun berdampak besar</b>, sehingga data kerugian historis "
            "sering tidak mencukupi untuk pemodelan statistik murni.", st["body"])]
        s += [Paragraph(
            "Pertanyaan yang dijawab model ini: <b>bagaimana mengukur risiko operasional "
            "secara <i>forward-looking</i> ketika data historis belum lengkap?</b> "
            "Pendekatan <i>scorecard</i> menjawabnya dengan memanfaatkan pengetahuan para "
            "ahli (self-assessment): sejumlah expert menilai frekuensi, severity, dan "
            "kualitas kontrol pada setiap kategori risiko, lalu penilaian itu diringkas "
            "menjadi sebuah <b>rating huruf</b> (mirip peringkat kredit) yang mudah dibaca "
            "manajemen dan dapat dibandingkan antar-kategori.", st["body"])]

        s += [heading("2. Landasan Teori &amp; Metodologi", st)]
        s += [Paragraph(
            "Model mengikuti Giudici (2009) Bab 12.4 (Gambar 12.1). Untuk setiap kategori "
            "risiko, distribusi opini expert diringkas menjadi rating melalui dua "
            "komponen: <b>lokasi</b> (seberapa tinggi tingkat risikonya) dan <b>konsensus</b> "
            "(seberapa sepakat para ahli).", st["body"])]
        s += [Paragraph("a) Lokasi &mdash; kelas median menentukan huruf dasar", st["h2"])]
        s += [Paragraph(
            "Setiap skala disusun dari risiko terendah ke tertinggi, dan setiap kelas "
            "diberi peringkat (rank). <b>Kelas median</b> dari seluruh suara expert dipakai "
            "untuk menetapkan huruf dasar: rank 1 = risiko terendah = <b>A</b>, lalu B, C, "
            "dan seterusnya. Untuk jumlah expert genap dipilih <i>upper median</i> "
            "(lebih konservatif terhadap risiko).", st["body"])]
        s += [Paragraph("b) Konsensus &mdash; indeks Gini ternormalisasi melipatkan huruf", st["h2"])]
        s += note_box(accent, st, "Rumus indeks Gini ternormalisasi", [
            Paragraph(
                f"{mono('G = 1 - jumlah(p_k^2)')} &nbsp; lalu dinormalkan &nbsp; "
                f"{mono('G_norm = G x K/(K-1)')}", st["note"]),
            Paragraph(
                "dengan <b>p_k</b> = proporsi suara pada kelas ke-k dan <b>K</b> = jumlah "
                "kelas. Nilai G_norm = 0 berarti semua expert sepakat satu kelas "
                "(konsensus penuh); G_norm = 1 berarti pendapat tersebar merata "
                "(heterogenitas maksimum).", st["note"]),
            Paragraph(
                "Aturan pelipatan huruf: G_norm tidak lebih dari 1/3 &rarr; <b>3 huruf</b> "
                "(mis. AAA); antara 1/3 dan 2/3 &rarr; <b>2 huruf</b> (AA); lebih dari 2/3 "
                "&rarr; <b>1 huruf</b> (A). Jadi konsensus tinggi memperkuat keyakinan "
                "pada rating.", st["note"]),
        ])
        s += [Paragraph("c) Perceived loss &mdash; estimasi kuantitatif tiap kategori", st["h2"])]
        s += [Paragraph(
            f"Untuk menghubungkan rating kualitatif ke angka, dihitung "
            f"{mono('perceived loss = frekuensi/tahun x nilai severity x faktor kontrol')}. "
            "Faktor kontrol (turunan sederhana dari faktor KRI pada buku) menaikkan "
            "ekspektasi kerugian saat kontrol buruk dan menurunkannya saat kontrol "
            "efektif. <b>Lampu lalu lintas</b> dipakai untuk visualisasi cepat: "
            "A = hijau, B = kuning, C dan di atasnya = merah.", st["body"])]
        s += [Paragraph("Glosarium istilah kunci:", st["h2"])]
        s += [glossary(accent, st, [
            ("Self-assessment", "Penilaian risiko oleh para ahli internal, bersifat forward-looking (memandang ke depan)."),
            ("Kelas median", "Kelas di tengah seluruh suara expert; penentu huruf dasar rating."),
            ("Indeks Gini", "Ukuran keberagaman pendapat; makin kecil makin sepakat (konsensus tinggi)."),
            ("Perceived loss", "Perkiraan kerugian moneter hasil terjemahan opini kualitatif ke angka."),
            ("Lampu lalu lintas", "Kode warna risiko: hijau (rendah), kuning (sedang), merah (tinggi)."),
        ])]

        s += [heading("3. Dataset &amp; Input", st)]
        s += [Paragraph(
            f"Input berupa kuesioner self-assessment sintetis: <b>{n_experts} expert</b> "
            f"menilai <b>{n_cat} kategori</b> Basel II, yaitu kombinasi 8 <i>business line</i> "
            "&times; 7 <i>event type</i>. Data dibuat setia pada metodologi karena data "
            "expert asli tidak tersedia, dan melengkapi data kerugian internal PaySim.",
            st["body"])]
        s += [Paragraph("Setiap penilaian memakai skala ordinal berikut:", st["body"])]
        s += bullets([
            "<b>Frekuensi</b> (4 kelas): yearly, monthly, weekly, daily &mdash; makin sering makin berisiko.",
            "<b>Severity</b> (7 kelas): irrelevant, minor, moderate, significant, major, severe, catastrophic.",
            "<b>Kontrol</b> (3 kelas): effective, to be adjusted, not effective.",
        ], st)

        s += [heading("4. Alur Proses Model", st), flow,
              Paragraph("Gambar 1. Alur scorecard: opini expert -> kelas median (huruf) -> "
                        "indeks Gini (pelipatan) -> rating + lampu lalu lintas.", st["caption"])]
        s += [Paragraph(
            "Secara prosedural: (1) kumpulkan suara expert per kategori; (2) untuk tiap "
            "dimensi hitung kelas median &rarr; huruf dasar; (3) hitung indeks Gini "
            "ternormalisasi &rarr; jumlah huruf; (4) gabungkan menjadi rating dan warna; "
            "(5) hitung perceived loss; (6) urutkan kategori menurut <i>priority score</i> "
            "(frekuensi &times; severity &times; kontrol) untuk menetapkan prioritas "
            "intervensi.", st["body"])]

        s += [heading("5. Contoh Perhitungan (Langkah demi Langkah)", st)]
        s += [Paragraph(
            f"Agar transparan, berikut perhitungan nyata dimensi <b>severity</b> pada "
            f"kategori prioritas teratas: <b>{we}</b> (lini {wb}). Delapan suara expert: "
            f"{mono(', '.join(sev_votes))}.", st["body"])]
        s += note_box(accent, st, "Penurunan rating severity kategori teratas", [
            Paragraph(
                f"<b>Langkah 1 &mdash; Kelas median (lokasi).</b> Peringkat 8 suara "
                f"diurutkan: {mono(', '.join(map(str, ranks_sorted)))}. Median atas (posisi "
                f"ke-{median_pos}) = peringkat {mono(str(sev_rank))} = kelas "
                f"'<b>{sev_median_class}</b>', dipetakan ke huruf dasar <b>{sev_letter}</b>.",
                st["step"]),
            Paragraph(
                f"<b>Langkah 2 &mdash; Indeks Gini (konsensus).</b> Proporsi tiap kelas: "
                f"{mono(counts_str)} dari {len(sev_votes)} suara. Maka "
                f"{mono('jumlah(p^2)')} = {mono(psq_str)} = {p_sq_sum:.3f}; "
                f"{mono('G = 1 - jumlah(p^2)')} = {gini_raw:.3f}. Normalisasi (K={K}): "
                f"{mono(f'G_norm = {gini_raw:.3f} x {K}/{K-1}')} = <b>{sev_gini:.3f}</b>.",
                st["step"]),
            Paragraph(
                f"<b>Langkah 3 &mdash; Aturan konsensus.</b> Karena G_norm = {sev_gini:.3f} "
                f"{rule_txt}, huruf {sev_letter} menjadi rating <b>{sev_rating}</b>.",
                st["step"]),
            Paragraph(
                f"<b>Cek silang.</b> Hasil manual <b>{sev_rating}</b> identik dengan "
                f"keluaran program ({top.iloc[0]['severity_rating']}). Langkah serupa "
                "diulang untuk dimensi frekuensi dan kontrol.", st["step"]),
        ])

        s += [heading("6. Hasil &amp; Analisis", st)]
        s += [metric_cards(accent, st, [
            (str(n_cat), "Kategori dinilai"),
            (str(n_experts), "Jumlah expert"),
            (rp(total_loss), "Total perceived loss"),
            (str(red_sev), "Kategori severity merah"),
        ]), Spacer(1, 10)]
        s += [Paragraph(
            "Total perkiraan kerugian seluruh kategori (self-assessment) mencapai "
            f"<b>{rp(total_loss)}</b>, dengan <b>{red_sev}</b> kategori berlabel severity "
            "merah (risiko dampak tinggi). Tabel berikut menampilkan 10 kategori dengan "
            "<i>priority score</i> tertinggi yang paling layak diprioritaskan untuk "
            "intervensi/penguatan kontrol.", st["body"])]
        s += [data_table(
            accent, st,
            ["#", "Business Line", "Event Type", "Frek", "Sev", "Ktrl",
             "Loss (jt)", "Skor"],
            top_rows,
            [0.7 * cm, 3.1 * cm, 3.9 * cm, 1.1 * cm, 1.1 * cm, 1.1 * cm,
             1.9 * cm, 1.1 * cm],
            aligns={0: "CENTER", 3: "CENTER", 4: "CENTER", 5: "CENTER",
                    6: "RIGHT", 7: "CENTER"}),
            Spacer(1, 4),
            Paragraph("Tabel 1. Sepuluh kategori prioritas intervensi tertinggi "
                      "(rating huruf per dimensi + perceived loss dalam juta).", st["caption"])]
        s += [chart1, Paragraph(
            "Gambar 2. Sepuluh kategori dengan perceived loss terbesar.", st["caption"])]
        s += [chart2, Paragraph(
            "Gambar 3. Sebaran rating lampu lalu lintas pada 56 kategori untuk tiga "
            "dimensi penilaian (frekuensi, severity, kontrol).", st["caption"])]

        s += [heading("7. Interpretasi &amp; Pembahasan", st)]
        s += [Paragraph(
            f"Kategori <b>{we}</b> pada lini <b>{wb}</b> menempati prioritas tertinggi "
            "karena mengombinasikan frekuensi tinggi, severity besar, dan kontrol yang "
            "lemah sekaligus. Pola pada Gambar 3 memperlihatkan bahwa dimensi severity "
            "menyumbang paling banyak label merah, menandakan bahwa <b>besarnya dampak</b> "
            "(bukan sekadar seringnya kejadian) menjadi pendorong utama profil risiko "
            "menurut pandangan para ahli. Rating huruf membuat hasil mudah dikomunikasikan: "
            "manajemen cukup melihat warna dan huruf untuk memahami kategori mana yang "
            "perlu tindakan.", st["body"])]

        s += [heading("8. Keterbatasan &amp; Saran Pengembangan", st)]
        s += bullets([
            "Opini expert bersifat <b>sintetis</b>; pada penerapan nyata perlu kuesioner "
            "aktual agar konsensus (Gini) mencerminkan keadaan sesungguhnya.",
            "Nilai severity moneter memakai titik-tengah kelas yang ditetapkan; sebaiknya "
            "dikalibrasi dengan data kerugian internal organisasi.",
            "Hasil scorecard sebaiknya tidak berdiri sendiri, melainkan digabung dengan "
            "data kerugian aktuaria dan eksternal (lihat Model 3) untuk gambaran utuh.",
        ], st)

        s += [heading("9. Kesimpulan", st)]
        s += [Paragraph(
            "Model scorecard berhasil menerjemahkan opini kualitatif para ahli menjadi "
            "rating risiko terstandar sekaligus estimasi kerugian kuantitatif untuk 56 "
            f"kategori Basel. Kategori <b>{we}</b> pada lini <b>{wb}</b> teridentifikasi "
            "sebagai prioritas intervensi tertinggi. Output scorecard ini menjadi titik "
            "data <i>self-assessment</i> yang kemudian diintegrasikan pada Model 3, "
            "sehingga melengkapi pandangan <i>backward-looking</i> dari Model aktuaria.",
            st["body"])]
        return s

    return build_pdf("Laporan_Model1_Scorecard_LisIndriani.pdf",
                     accent, "Model 1 Scorecard", member, story)


# ===========================================================================
# MODEL 2 — VaR AKTUARIA  (Ana Sulistiana Alwi)
# ===========================================================================
def report_actuarial():
    accent = "#2563c9"  # biru
    st = styles(accent)
    member = "Ana Sulistiana Alwi"

    internal = load_internal_losses()
    fs = internal_frequency_severity(internal)
    freq_daily = fs["n_events"] / fs["span_days"]
    summ = actuarial_summary(freq_daily, fs["severity_sample"])
    param = summ["_param_losses"]
    emp = summ["_empirical_losses"]
    hist_var = summ["historical_var"]
    act_var = summ["actuarial_montecarlo_var"]
    mu, sigma = summ["severity_mu"], summ["severity_sigma"]

    sev_pos = fs["severity_sample"][fs["severity_sample"] > 0]
    mean_sev = float(sev_pos.mean())
    z999 = NormalDist().inv_cdf(0.999)
    logn_q = float(np.exp(mu + sigma * z999))
    el_manual = freq_daily * mean_sev
    n_events = int(fs["n_events"])
    span_days = int(fs["span_days"])
    lam_formula = f"lambda = n_events / span = {n_events:,} / {span_days}"
    el_formula = f"E[L] = lambda x rata-rata(X) = {freq_daily:,.1f} x {mean_sev:,.0f}"
    logn_formula = f"exp(mu + sigma x z) = exp({mu:.3f} + {sigma:.3f} x {z999:.3f})"

    # --- Chart 1: dua panel distribusi kerugian agregat ---
    fig1, (axa, axb) = plt.subplots(1, 2, figsize=(7.8, 3.2))
    e = emp[emp <= hist_var * 1.8]
    axa.hist(e / 1e6, bins=40, color="#7aa7ec", edgecolor="#2563c9", alpha=0.9)
    axa.axvline(hist_var / 1e6, color="#d62728", lw=1.8,
                label=f"Hist. VaR 99.9%\n{hist_var/1e6:,.0f} jt")
    axa.set_title("Severity Empiris (historis)")
    axa.set_xlabel("Kerugian agregat / hari (juta Rp)")
    axa.set_ylabel("Frekuensi simulasi")
    axa.legend(fontsize=7, frameon=False)
    p = param[param <= act_var * 1.25]
    axb.hist(p / 1e6, bins=40, color="#9bd0a6", edgecolor="#1f9d55", alpha=0.9)
    axb.axvline(act_var / 1e6, color="#d62728", lw=1.8,
                label=f"Aktuaria MC VaR 99.9%\n{act_var/1e6:,.0f} jt")
    axb.set_title("Severity Lognormal (parametrik)")
    axb.set_xlabel("Kerugian agregat / hari (juta Rp)")
    axb.legend(fontsize=7, frameon=False)
    fig1.suptitle("Distribusi Kerugian Agregat L = Sigma X (Monte Carlo, 20.000 simulasi)",
                  fontsize=10, fontweight="bold")
    fig1.tight_layout(rect=(0, 0, 1, 0.94))
    chart1 = mpl_image(fig1)

    # --- Chart 2: perbandingan E[L] vs dua VaR ---
    fig2, ax2 = plt.subplots(figsize=(7.6, 2.8))
    names = ["Kerugian\nterealisasi E[L]", "Historical VaR\n99.9%",
             "Aktuaria MC VaR\n99.9% (lognormal)"]
    vals = [summ["historical_losses"] / 1e6, hist_var / 1e6, act_var / 1e6]
    bars = ax2.bar(names, vals, color=["#9aa6b2", "#2563c9", "#1f9d55"],
                   edgecolor="#333")
    ax2.set_ylabel("Juta Rp")
    ax2.set_title("Ekspektasi vs Value at Risk (basis harian)")
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")
    chart2 = mpl_image(fig2)

    flow = flow_diagram([
        ("INPUT", "Kerugian fraud\nPaySim (internal)"),
        ("FREKUENSI", "N ~ Poisson\n(lambda/hari)"),
        ("SEVERITY", "X ~ Lognormal\n(mu, sigma)"),
        ("MONTE CARLO", "L = Sigma X ->\nVaR 99.9%"),
    ], accent)

    def story():
        s = []
        s += title_band(
            accent, st,
            "LAPORAN MODEL 2 - PENGUKURAN RISIKO OPERASIONAL",
            "Model Aktuaria (Value at Risk)",
            "Penanggung jawab: <b>Ana Sulistiana Alwi</b>",
            [("Modul kode", "orm/actuarial.py"),
             ("Acuan buku", "Giudici 2009, Gbr 12.2"),
             ("Tanggal", datetime.now().strftime("%d %b %Y"))])

        s += [heading("1. Latar Belakang &amp; Rumusan Masalah", st)]
        s += [Paragraph(
            "Regulator (Basel II) mensyaratkan bank menyediakan modal untuk menutup "
            "potensi kerugian operasional. Salah satu pendekatan lanjutan adalah "
            "memodelkan kerugian secara statistik dari <b>data kerugian internal</b> yang "
            "benar-benar pernah terjadi (<i>backward-looking</i>), lalu menghitung "
            "kebutuhan modal sebagai kuantil tinggi dari distribusi kerugian.", st["body"])]
        s += [Paragraph(
            "Rumusan masalah: <b>berapa potensi kerugian operasional maksimum dalam satu "
            "periode pada tingkat keyakinan 99,9%?</b> Ukuran ini disebut "
            "<b>Value at Risk (VaR)</b> &mdash; ambang kerugian yang hanya akan terlampaui "
            "sekali dalam seribu periode.", st["body"])]

        s += [heading("2. Landasan Teori &amp; Metodologi", st)]
        s += [Paragraph(
            "Model mengikuti Giudici (2009) Bab 12.3 (Gambar 12.2) dengan kerangka "
            "<i>loss distribution approach</i>. Total kerugian satu periode adalah "
            "penjumlahan sejumlah <b>acak</b> impak kerugian:", st["body"])]
        s += note_box(accent, st, "Model frekuensi-severity (aktuaria)", [
            Paragraph(f"{mono('L = X_1 + X_2 + ... + X_N')}", st["note"]),
            Paragraph(
                "dengan <b>N</b> = banyaknya kejadian (frekuensi) dan <b>X_i</b> = besar "
                "kerugian tiap kejadian (severity). Keduanya dimodelkan terpisah lalu "
                "digabung:", st["note"]),
            Paragraph(
                f"&bull; Frekuensi: {mono('N ~ Poisson(lambda)')}, dengan lambda = "
                "rata-rata kejadian per hari.", st["note"]),
            Paragraph(
                f"&bull; Severity: {mono('X ~ Lognormal(mu, sigma)')}, di-fit dari "
                "logaritma kerugian nyata.", st["note"]),
            Paragraph(
                "&bull; Konvolusi keduanya dilakukan dengan <b>simulasi Monte Carlo</b> "
                "(20.000 skenario) untuk membentuk distribusi L, lalu "
                f"{mono('VaR 99.9% = persentil ke-99,9 dari L')}.", st["note"]),
        ])
        s += [Paragraph(
            "Dihitung dua varian VaR: <b>historis</b> (severity di-<i>resample</i> langsung "
            "dari data, tidak mengasumsikan bentuk distribusi) dan <b>parametrik</b> "
            "(severity mengikuti lognormal). Perbedaan keduanya mengukur "
            "<b>risiko model</b>.", st["body"])]
        s += [Paragraph("Glosarium istilah kunci:", st["h2"])]
        s += [glossary(accent, st, [
            ("Frekuensi (N)", "Berapa kali kejadian kerugian muncul dalam satu periode."),
            ("Severity (X)", "Besarnya kerugian per satu kejadian."),
            ("Distribusi Poisson", "Model peluang untuk mencacah kejadian acak yang jarang per satuan waktu."),
            ("Distribusi Lognormal", "Model severity bernilai positif dengan ekor kanan panjang (kejadian ekstrem mungkin)."),
            ("Monte Carlo", "Simulasi ribuan skenario acak untuk memperkirakan distribusi hasil."),
            ("VaR 99.9%", "Ambang kerugian yang hanya terlampaui 0,1% periode (1 dari 1000)."),
        ])]

        s += [heading("3. Dataset &amp; Input", st)]
        s += [Paragraph(
            f"Sumber data internal: <b>{fs['n_events']:,} kejadian</b> kerugian fraud dari "
            f"dataset PaySim selama <b>{fs['span_days']} hari</b> pengamatan, dipetakan ke "
            "kategori <i>Payment and Settlement / External Fraud</i> (sesuai pemetaan "
            f"satu area pada buku). Total kerugian internal tercatat "
            f"<b>{rp(fs['total_internal'])}</b>. Periode analisis ditetapkan harian agar "
            "simulasi ringan dan konsisten dengan model lain.", st["body"])]

        s += [heading("4. Alur Proses Model", st), flow,
              Paragraph("Gambar 1. Alur aktuaria: data internal -> fit Poisson (frekuensi) "
                        "dan Lognormal (severity) -> Monte Carlo -> VaR 99.9%.", st["caption"])]

        s += [heading("5. Contoh Perhitungan (Langkah demi Langkah)", st)]
        s += note_box(accent, st, "Penurunan angka dari data nyata", [
            Paragraph(
                f"<b>Langkah 1 &mdash; Frekuensi.</b> "
                f"{mono(lam_formula)} = <b>{freq_daily:,.1f}</b> kejadian/hari.", st["step"]),
            Paragraph(
                f"<b>Langkah 2 &mdash; Fit severity lognormal.</b> Dari logaritma kerugian: "
                f"{mono(f'mu = rata-rata(ln X) = {mu:.3f}')}, "
                f"{mono(f'sigma = simpangan baku(ln X) = {sigma:.3f}')}.", st["step"]),
            Paragraph(
                f"<b>Langkah 3 &mdash; Kerugian terekspektasi.</b> "
                f"{mono(el_formula)} = <b>{rp(el_manual)}</b> (sesuai keluaran program "
                f"{rp(summ['historical_losses'])}).", st["step"]),
            Paragraph(
                f"<b>Langkah 4 &mdash; Severity ekstrem (P99.9).</b> Parametrik: "
                f"{mono(logn_formula)} = <b>{rp(logn_q)}</b>, jauh di atas empiris "
                f"{rp(summ['severity_empirical_q999'])} &mdash; inilah sumber risiko model.",
                st["step"]),
            Paragraph(
                f"<b>Langkah 5 &mdash; VaR agregat.</b> Monte Carlo menggabung frekuensi "
                "&times; severity 20.000 kali, lalu ambil persentil 99,9: Historical VaR "
                f"= <b>{rp(hist_var)}</b>, Aktuaria MC VaR = <b>{rp(act_var)}</b>.",
                st["step"]),
        ])

        s += [heading("6. Hasil &amp; Analisis", st)]
        s += [metric_cards(accent, st, [
            (f"{freq_daily:,.1f}", "Frekuensi/hari (lambda)"),
            (rp(summ["historical_losses"]), "Kerugian E[L]/hari"),
            (rp(hist_var), "Historical VaR 99.9%"),
            (rp(act_var), "Aktuaria MC VaR 99.9%"),
        ]), Spacer(1, 10)]
        s += [Paragraph(
            f"Frekuensi terukur <b>{freq_daily:,.1f} kejadian/hari</b> dengan severity "
            f"lognormal (mu={mu:.3f}, sigma={sigma:.3f}). VaR historis 99,9% sebesar "
            f"<b>{rp(hist_var)}</b>, sedangkan VaR parametrik jauh lebih besar "
            f"(<b>{rp(act_var)}</b>) karena ekor lognormal yang tebal (sigma tinggi) "
            "memperhitungkan kejadian ekstrem yang belum pernah teramati pada data "
            "historis terbatas.", st["body"])]
        s += [chart2, Paragraph(
            "Gambar 2. Perbandingan kerugian terealisasi (E[L]) dengan dua estimasi VaR.",
            st["caption"])]
        s += [chart1, Paragraph(
            "Gambar 3. Distribusi kerugian agregat hasil Monte Carlo: empiris (kiri) "
            "vs parametrik lognormal (kanan), beserta garis VaR 99.9%.", st["caption"])]
        s += [data_table(
            accent, st,
            ["Diagnostik ekor severity (risiko model)", "Nilai"],
            [["Sigma lognormal", f"{sigma:.3f}"],
             ["Severity maksimum teramati", rp(summ["severity_max_observed"])],
             ["Severity empiris persentil 99.9", rp(summ["severity_empirical_q999"])],
             ["Severity lognormal persentil 99.9", rp(summ["severity_lognormal_q999"])]],
            [10.5 * cm, 5.7 * cm], aligns={1: "RIGHT"}),
            Spacer(1, 4),
            Paragraph("Tabel 1. Diagnostik ekor: ekor parametrik dapat jauh melampaui "
                      "nilai maksimum yang pernah teramati.", st["caption"])]

        s += [heading("7. Interpretasi &amp; Pembahasan", st)]
        s += [Paragraph(
            "Selisih besar antara VaR historis dan parametrik adalah temuan terpenting. "
            "VaR historis hanya 'mengingat' kerugian yang pernah terjadi, sehingga "
            "cenderung <b>meremehkan</b> kejadian ekstrem yang belum tercatat. Sebaliknya, "
            "lognormal dengan sigma tinggi menghasilkan ekor sangat panjang, sehingga VaR "
            "parametrik jauh lebih besar. Pemilihan distribusi severity karenanya sangat "
            "menentukan besaran modal risiko &mdash; keputusan yang harus diambil hati-hati "
            "dan didukung uji kecocokan distribusi.", st["body"])]

        s += [heading("8. Keterbatasan &amp; Saran Pengembangan", st)]
        s += bullets([
            "Asumsi lognormal pada severity belum diuji formal; sebaiknya dibandingkan "
            "dengan distribusi ekor lain (mis. GPD / extreme value theory).",
            "Severity PaySim memiliki batas atas (pemotongan), sehingga ekor empiris bisa "
            "terlalu pendek; pada data nyata perlu data kerugian besar yang memadai.",
            "Independensi antar-kejadian diasumsikan; korelasi/penggugusan kejadian dapat "
            "menambah ketebalan ekor dan perlu dipertimbangkan.",
        ], st)

        s += [heading("9. Kesimpulan", st)]
        s += [Paragraph(
            "Model aktuaria memberikan ukuran VaR 99,9% yang terkalibrasi pada data "
            "kerugian nyata melalui kerangka frekuensi-severity dan simulasi Monte Carlo. "
            "Perbedaan mencolok antara VaR historis dan parametrik menegaskan adanya "
            "<b>risiko model</b> pada pemodelan ekor. VaR aktuaria ini menjadi salah satu "
            "metrik pembanding utama yang diintegrasikan pada Model 3.", st["body"])]
        return s

    return build_pdf("Laporan_Model2_Aktuaria_AnaSulistianaAlwi.pdf",
                     accent, "Model 2 Aktuaria", member, story)


# ===========================================================================
# MODEL 3 — VaR INTEGRASI BAYESIAN + BIA  (Andi Agung Dwi Arya B)
# ===========================================================================
def report_integrated():
    accent = "#7a3ea8"  # ungu
    st = styles(accent)
    member = "Andi Agung Dwi Arya B"

    res = run_comparison(seed=42)
    comp = res["comparison"]
    gross_income = res["gross_income"]
    freq = res["frequency"]
    sa_point = res["self_assessment_point"]
    total_internal = res["internal_summary"]["total_internal"]

    # Rekonstruksi nilai antara untuk contoh perhitungan (seed identik)
    internal = load_internal_losses()
    fs = internal_frequency_severity(internal)
    sev = fs["severity_sample"]
    external = generate_external_losses(internal_severity=sev, seed=42)
    total_external = float(external.sum())
    constant = total_external / fs["total_internal"]
    n_internal = int((sev > 0).sum())
    n_external = int(len(external))
    n_integrated = n_internal + 1 + n_external

    order = [
        "Historical losses", "Self-assessment VaR", "Bayes VaR",
        "Basic Indicator Approach VaR", "Historical VaR",
        "Bayes Montecarlo VaR", "Actuarial Montecarlo VaR",
    ]
    comp = comp.reindex(order)
    bayes_simple = comp["Bayes VaR"]
    bayes_mc = comp["Bayes Montecarlo VaR"]
    bia = comp["Basic Indicator Approach VaR"]

    # --- Chart 1: perbandingan 7 VaR (analog Gambar 12.2) ---
    fig1, ax1 = plt.subplots(figsize=(7.8, 3.8))
    vals = comp.to_numpy() / 1e6
    palette = ["#9aa6b2", "#c39bd3", "#a569bd", "#7a3ea8",
               "#5b2c82", "#6c5ce7", "#2d6cdf"]
    ax1.barh(range(len(comp)), vals, color=palette, edgecolor="#333")
    ax1.set_yticks(range(len(comp)))
    ax1.set_yticklabels(comp.index, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("Value at Risk (juta Rp)")
    ax1.set_title("Perbandingan 7 Metode Pengukuran VaR (analog Gambar 12.2)")
    for i, v in enumerate(vals):
        ax1.text(v, i, f" {v:,.0f}", va="center", fontsize=7.5)
    chart1 = mpl_image(fig1)

    # --- Chart 2: tiga filosofi (internal / forward / integrasi) ---
    fig2, ax2 = plt.subplots(figsize=(7.6, 2.9))
    names = ["Aktuaria MC\n(internal saja)", "Self-Assessment\n(forward-looking)",
             "Bayes MC\n(integrasi 3 sumber)"]
    triad = [comp["Actuarial Montecarlo VaR"] / 1e6,
             comp["Self-assessment VaR"] / 1e6,
             comp["Bayes Montecarlo VaR"] / 1e6]
    bars = ax2.bar(names, triad, color=["#2d6cdf", "#1f9d55", "#7a3ea8"],
                   edgecolor="#333")
    ax2.set_ylabel("Juta Rp")
    ax2.set_title("Efek Integrasi: VaR Tunggal vs VaR Gabungan")
    for b, v in zip(bars, triad):
        ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")
    chart2 = mpl_image(fig2)

    flow = flow_diagram([
        ("3 SUMBER DATA", "internal +\nself-assess + eksternal"),
        ("SCALING", "samakan skala\nantar sumber"),
        ("INTEGRASI", "distribusi kerugian\nterintegrasi (Bayes)"),
        ("OUTPUT", "VaR integrasi\n+ BIA 15%"),
    ], accent)

    comp_rows = [[name, rp(val)] for name, val in comp.items()]

    def story():
        s = []
        s += title_band(
            accent, st,
            "LAPORAN MODEL 3 - PENGUKURAN RISIKO OPERASIONAL",
            "Model Terintegrasi Bayesian + BIA",
            "Penanggung jawab: <b>Andi Agung Dwi Arya B</b>",
            [("Modul kode", "orm/integrated.py"),
             ("Acuan buku", "Giudici 2009, Gbr 12.2"),
             ("Tanggal", datetime.now().strftime("%d %b %Y"))])

        s += [heading("1. Latar Belakang &amp; Rumusan Masalah", st)]
        s += [Paragraph(
            "Setiap model tunggal punya kelemahan. Model aktuaria hanya memakai kerugian "
            "internal (<i>backward-looking</i>), sehingga buta terhadap risiko yang belum "
            "pernah terjadi. Scorecard hanya memakai opini ahli (<i>forward-looking</i>), "
            "sehingga subjektif. Data kerugian eksternal industri menambah pengalaman bank "
            "lain, tetapi berskala berbeda.", st["body"])]
        s += [Paragraph(
            "Rumusan masalah: <b>bagaimana menggabungkan ketiga sumber data menjadi satu "
            "ukuran VaR yang lebih lengkap, sekaligus membandingkannya dengan tolok ukur "
            "regulator?</b> Pendekatan Bayesian menyatukannya, dan <b>Basic Indicator "
            "Approach (BIA)</b> Basel II dipakai sebagai pembanding top-down.", st["body"])]

        s += [heading("2. Landasan Teori &amp; Metodologi", st)]
        s += [Paragraph(
            "Model mengikuti Giudici (2009) Bab 12.4 (Gambar 12.2). Tiga aliran data "
            "disatukan menjadi satu distribusi kerugian terintegrasi, lalu VaR diturunkan "
            "darinya. Karena skala antar-sumber berbeda, dilakukan <b>scaling</b> "
            "lebih dulu.", st["body"])]
        s += note_box(accent, st, "Scaling, integrasi, dan BIA", [
            Paragraph(
                f"&bull; <b>Scaling eksternal:</b> {mono('c = total_eksternal / total_internal')}; "
                "setiap kerugian eksternal dibagi c agar sebanding dengan skala internal.",
                st["note"]),
            Paragraph(
                "&bull; <b>Distribusi terintegrasi:</b> gabungan dari seluruh kerugian "
                "internal + satu titik self-assessment (biasanya lebih tinggi dari kerugian "
                "aktual) + kerugian eksternal yang sudah di-scaling.", st["note"]),
            Paragraph(
                f"&bull; <b>Bayes VaR (simple):</b> {mono('persentil 99,9% dari distribusi terintegrasi')}. "
                "<b>Bayes MC VaR:</b> simulasi agregat Monte Carlo berbasis kerugian "
                "terintegrasi (paralel dengan aktuaria).", st["note"]),
            Paragraph(
                f"&bull; <b>Basic Indicator Approach (BIA):</b> {mono('VaR = 15% x gross income')} "
                "&mdash; tolok ukur regulator yang dihitung dari indikator pendapatan, "
                "bukan dari data kerugian.", st["note"]),
        ])
        s += [Paragraph("Glosarium istilah kunci:", st["h2"])]
        s += [glossary(accent, st, [
            ("Backward-looking", "Berbasis kejadian masa lalu (data kerugian internal)."),
            ("Forward-looking", "Berbasis perkiraan ke depan (opini self-assessment)."),
            ("Scaling", "Penyamaan skala data eksternal agar sebanding dengan internal."),
            ("Integrasi Bayesian", "Penggabungan beberapa sumber bukti menjadi satu distribusi kerugian."),
            ("BIA", "Basic Indicator Approach: modal risiko = 15% dari gross income (Basel II)."),
            ("Gross income", "Indikator pendapatan kotor sebagai dasar perhitungan BIA."),
        ])]

        s += [heading("3. Dataset &amp; Input", st)]
        s += [Paragraph(
            "Menggabungkan tiga sumber: (1) kerugian internal PaySim, (2) satu titik "
            "self-assessment dari scorecard 56 kategori (Model 1), dan (3) kerugian "
            f"eksternal sintetis terpool di atas ambang tertentu. Frekuensi internal "
            f"<b>{freq:,.1f} kejadian/hari</b>; gross income ilustratif untuk BIA "
            f"<b>{rp(gross_income)}</b>. Semua metrik memakai basis periode sama (harian) "
            "agar urutan antar-metode konsisten.", st["body"])]

        s += [heading("4. Alur Proses Model", st), flow,
              Paragraph("Gambar 1. Alur integrasi: tiga sumber data -> scaling -> "
                        "distribusi kerugian terintegrasi -> VaR Bayesian + BIA.",
                        st["caption"])]

        s += [heading("5. Contoh Perhitungan (Langkah demi Langkah)", st)]
        s += note_box(accent, st, "Penurunan angka dari data nyata", [
            Paragraph(
                f"<b>Langkah 1 &mdash; Kumpulkan tiga sumber.</b> Internal "
                f"{n_internal:,} titik kerugian + 1 titik self-assessment "
                f"({rp(sa_point)}) + eksternal {n_external:,} titik.", st["step"]),
            Paragraph(
                f"<b>Langkah 2 &mdash; Scaling eksternal.</b> "
                f"{mono(f'c = {total_external:,.0f} / {total_internal:,.0f}')} "
                f"= <b>{constant:.3f}</b>; tiap kerugian eksternal dibagi {constant:.3f} "
                "agar sebanding dengan internal.", st["step"]),
            Paragraph(
                f"<b>Langkah 3 &mdash; Integrasi.</b> Gabungan menjadi "
                f"{mono(f'{n_internal:,} + 1 + {n_external:,} = {n_integrated:,}')} titik "
                f"kerugian. Bayes VaR = persentil 99,9% gabungan = <b>{rp(bayes_simple)}</b>.",
                st["step"]),
            Paragraph(
                f"<b>Langkah 4 &mdash; BIA.</b> "
                f"{mono(f'VaR = 0,15 x {gross_income:,.0f}')} = <b>{rp(bia)}</b>. "
                "Gross income di sini disetel ilustratif sehingga BIA setara Bayes VaR "
                "(meniru pola Gambar 12.2 buku).", st["step"]),
            Paragraph(
                f"<b>Langkah 5 &mdash; Bayes Monte Carlo.</b> Simulasi agregat berbasis "
                f"kerugian terintegrasi menghasilkan Bayes MC VaR = <b>{rp(bayes_mc)}</b>.",
                st["step"]),
        ])

        s += [heading("6. Hasil &amp; Analisis", st)]
        s += [metric_cards(accent, st, [
            (rp(bayes_simple), "Bayes VaR (simple)"),
            (rp(bayes_mc), "Bayes MC VaR"),
            (rp(bia), "BIA VaR (15%)"),
            (rp(comp["Self-assessment VaR"]), "Self-assessment VaR"),
        ]), Spacer(1, 10)]
        s += [Paragraph(
            "Tabel berikut merangkum tujuh metrik VaR pada basis periode yang sama "
            "(harian), analog Gambar 12.2 buku. Yang dibandingkan adalah <b>urutan/relasi "
            "antar-metode</b>, bukan besaran absolut (skala mengikuti karakteristik "
            "data PaySim).", st["body"])]
        s += [data_table(
            accent, st,
            ["Metode pengukuran", "Value at Risk"],
            comp_rows, [10.5 * cm, 5.7 * cm], aligns={1: "RIGHT"}),
            Spacer(1, 4),
            Paragraph("Tabel 1. Tujuh metode VaR (analog Gambar 12.2, basis harian).",
                      st["caption"])]
        s += [chart1, Paragraph(
            "Gambar 2. Perbandingan tujuh metode pengukuran VaR.", st["caption"])]
        s += [chart2, Paragraph(
            "Gambar 3. Efek integrasi: VaR dari sumber tunggal vs VaR gabungan tiga sumber.",
            st["caption"])]

        s += [heading("7. Interpretasi &amp; Pembahasan", st)]
        s += [Paragraph(
            "Urutan VaR mencerminkan asumsi tiap metode. <b>Actuarial Montecarlo VaR</b> "
            "tertinggi karena ekor lognormal yang tebal; <b>Bayes Montecarlo VaR</b> lebih "
            "rendah karena penambahan data eksternal dan self-assessment menstabilkan ekor. "
            "<b>Bayes VaR (simple)</b> dan <b>BIA</b> berada di level pembanding bawah. "
            "Pesan utamanya: integrasi tidak sekadar menjumlah, melainkan "
            "<b>menyeimbangkan</b> pandangan masa lalu, perkiraan ke depan, dan pengalaman "
            "industri sehingga hasilnya lebih kokoh terhadap keterbatasan satu sumber.",
            st["body"])]

        s += [heading("8. Keterbatasan &amp; Saran Pengembangan", st)]
        s += bullets([
            "Data eksternal dan self-assessment bersifat <b>sintetis</b>; integrasi nyata "
            "memerlukan data konsorsium (mis. DIPO) dan kuesioner aktual.",
            "Gross income BIA di sini ilustratif; pada penerapan nyata diisi angka "
            "keuangan organisasi yang sebenarnya.",
            "Pembobotan Bayesian dapat dikembangkan lebih formal (prior-posterior) untuk "
            "menimbang kredibilitas tiap sumber data secara eksplisit.",
        ], st)

        s += [heading("9. Kesimpulan", st)]
        s += [Paragraph(
            "Model terintegrasi menyatukan pandangan <i>backward-looking</i> (aktuaria), "
            "<i>forward-looking</i> (self-assessment), dan pengalaman industri (eksternal) "
            "ke dalam satu kerangka VaR, lalu membandingkannya dengan tolok ukur regulator "
            "(BIA). Hasilnya memberi manajemen gambaran risiko yang lebih utuh dan "
            "konsisten antar-metode, sesuai rekomendasi Giudici (2009) Bab 12. Bersama "
            "Model 1 dan Model 2, laporan ini melengkapi tiga sudut pandang pengukuran "
            "risiko operasional.", st["body"])]
        return s

    return build_pdf("Laporan_Model3_IntegrasiBIA_AndiAgungDwiAryaB.pdf",
                     accent, "Model 3 Integrasi", member, story)


def main():
    print("Membuat laporan PDF per model (menjalankan modul orm/) ...\n")
    paths = [report_scorecard(), report_actuarial(), report_integrated()]
    print("Selesai. File PDF yang dihasilkan:")
    for p in paths:
        size_kb = p.stat().st_size / 1024
        print(f"  - {p.name}  ({size_kb:,.0f} KB)")


if __name__ == "__main__":
    main()
