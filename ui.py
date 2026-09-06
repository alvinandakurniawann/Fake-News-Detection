# ui.py
"""
Design system terpusat untuk Fake News Detector — Midnight Analyst.
Satu sumber CSS + komponen presentasi. Logika deteksi tidak disentuh.
"""

import html as _html

import streamlit as st


TEAL = "#22D3EE"
TEAL_INK = "#A5F3FC"
RUST = "#F97316"
RUST_INK = "#FDBA74"
INK = "#E8EEF4"
MUTED = "#93A1B5"
LINE = "#22304A"
SURFACE = "#101828"
SURFACE_2 = "#162032"

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body {
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
/* Warisi Inter untuk kontrol form, TAPI jangan sentuh span ikon:
   Streamlit memakai ligatur Material Symbols (mis. panah dropdown),
   yang rusak menjadi teks mentah bila font-nya ditimpa. */
button, input, textarea, select {
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
span[class*="material-symbols"], span[class*="MaterialSymbols"],
i[class*="material-symbols"], [data-testid*="Icon"] span {
    font-family: 'Material Symbols Rounded', 'Material Symbols', sans-serif !important;
}
code, .mono, [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
    font-variant-numeric: tabular-nums;
}

/* Hero */
.fnd-hero { padding: 6px 2px 2px; }
.fnd-kicker {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.16em;
    text-transform: uppercase; color: #22D3EE; margin-bottom: 6px;
    display: flex; align-items: center; gap: 8px;
}
.fnd-kicker::before { content: ""; width: 20px; height: 2px; background: #22D3EE; border-radius: 1px; box-shadow: 0 0 8px rgba(34,211,238,0.8); }
.fnd-title { font-size: 2rem; font-weight: 800; letter-spacing: -0.025em; line-height: 1.12; margin: 0 0 8px; color: #F2F6FB; }
.fnd-lede { font-size: 1rem; color: #AEBACB; max-width: 78ch; margin: 0 0 14px; }
.fnd-steps { display: flex; gap: 8px; flex-wrap: wrap; margin: 0 0 4px; padding: 0; list-style: none; }
.fnd-steps li {
    display: flex; align-items: center; gap: 8px; font-size: 0.83rem; font-weight: 600;
    color: #D6DEE9; background: #101828; border: 1px solid #22304A;
    border-radius: 8px; padding: 7px 12px;
}
.fnd-steps .n {
    width: 20px; height: 20px; display: inline-grid; place-items: center;
    background: #22D3EE; color: #062A33; border-radius: 6px;
    font-size: 0.7rem; font-weight: 700; font-family: 'IBM Plex Mono', monospace;
}

/* Vonis */
.fnd-verdict { border-radius: 10px; padding: 16px 18px; margin: 14px 0; border: 1px solid; }
.fnd-verdict .v-label { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; opacity: 0.7; }
.fnd-verdict .v-name { font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em; margin: 2px 0; }
.fnd-verdict .v-conf { font-family: 'IBM Plex Mono', monospace; font-size: 0.9rem; font-weight: 600; }
.fnd-verdict .v-note { font-size: 0.83rem; margin-top: 6px; opacity: 0.8; }
.fnd-verdict.fake { background: #231009; border-color: #7C2D12; color: #FDBA74; }
.fnd-verdict.real { background: #062A30; border-color: #0E7490; color: #A5F3FC; }

/* Meter keyakinan */
.fnd-meter { background: #101828; border: 1px solid #22304A; border-radius: 10px; padding: 14px 18px 16px; margin: 12px 0; }
.fnd-meter .m-top { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; }
.fnd-meter .m-label { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #93A1B5; }
.fnd-meter .m-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 600; color: #F2F6FB; }
.fnd-meter .m-track { position: relative; height: 12px; border-radius: 7px; background: #1B2740; overflow: visible; }
.fnd-meter .m-fill { height: 100%; border-radius: 7px; background: #22D3EE; box-shadow: 0 0 12px rgba(34,211,238,0.45); transition: width 0.6s ease; }
.fnd-meter .m-tick { position: absolute; top: -4px; bottom: -4px; width: 2px; background: #5B6B82; border-radius: 1px; }
.fnd-meter .m-scale { display: flex; justify-content: space-between; font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #5B6B82; margin-top: 5px; }

/* Bar probabilitas */
.fnd-prob { background: #101828; border: 1px solid #22304A; border-radius: 10px; padding: 14px 18px; margin: 12px 0; display: grid; gap: 12px; }
.fnd-prob .p-row { display: grid; grid-template-columns: 110px 1fr 52px; gap: 10px; align-items: center; }
.fnd-prob .p-name { font-size: 0.83rem; font-weight: 650; color: #D6DEE9; }
.fnd-prob .p-track { height: 14px; border-radius: 7px; background: #1B2740; overflow: hidden; }
.fnd-prob .p-fill { height: 100%; border-radius: 7px; transition: width 0.6s ease; }
.fnd-prob .p-fill.fake { background: #F97316; }
.fnd-prob .p-fill.real { background: #22D3EE; }
.fnd-prob .p-val { font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; font-weight: 600; color: #F2F6FB; text-align: right; }

/* Kotak bacaan */
.fnd-box {
    border: 1px solid #22304A; border-radius: 10px; background: #101828;
    padding: 14px 16px; max-height: 320px; overflow-y: auto;
    line-height: 1.7; font-size: 0.93rem; color: #DCE4EE; white-space: pre-wrap;
}
.fnd-box.pre { font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; background: #0D1526; }
.fnd-meta { font-size: 0.8rem; color: #93A1B5; margin: 2px 0 10px; }
.fnd-meta b { color: #E8EEF4; }

/* Highlight kata — tint bertingkat di atas navy */
.hl { padding: 1px 4px; border-radius: 4px; margin: 0 1px; font-weight: 600; cursor: help; }
.hl-f1 { background: rgba(249,115,22,0.20); color: #FDBA74; border-bottom: 2px solid rgba(249,115,22,0.6); }
.hl-f2 { background: rgba(249,115,22,0.34); color: #FED7AA; border-bottom: 2px solid rgba(249,115,22,0.85); }
.hl-f3 { background: #C2410C; color: #fff; }
.hl-r1 { background: rgba(34,211,238,0.15); color: #A5F3FC; border-bottom: 2px solid rgba(34,211,238,0.55); }
.hl-r2 { background: rgba(34,211,238,0.28); color: #CFFAFE; border-bottom: 2px solid rgba(34,211,238,0.8); }
.hl-r3 { background: #0E7490; color: #fff; }
.hl-legend { display: flex; gap: 16px; font-size: 0.83rem; color: #93A1B5; margin-top: 10px; }
.hl-legend i { display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: 6px; }

/* Lencana prediksi */
.badge { display: inline-block; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.06em; border-radius: 6px; padding: 3px 10px; }
.badge-fake { background: #231009; color: #FDBA74; border: 1px solid #7C2D12; }
.badge-real { background: #062A30; color: #A5F3FC; border: 1px solid #0E7490; }

/* Chip domain */
.domains { display: flex; gap: 7px; flex-wrap: wrap; margin: 6px 0 0; padding: 0; list-style: none; }
.domains li { font-size: 0.76rem; font-weight: 600; color: #C6D2E2; background: #101828; border: 1px solid #22304A; border-radius: 999px; padding: 3px 11px; }

/* Section */
.fnd-section { font-size: 1.02rem; font-weight: 750; letter-spacing: -0.01em; color: #F2F6FB; margin: 0 0 4px; }
.fnd-sub { font-size: 0.85rem; color: #93A1B5; margin: 0 0 12px; }

@media (prefers-reduced-motion: reduce) {
    .fnd-meter .m-fill, .fnd-prob .p-fill { transition: none; }
}
</style>
"""

_PLOTLY_DARK = {
    "font": {"family": "Inter, system-ui, sans-serif", "color": "#C7D2E0", "size": 12},
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "margin": {"l": 8, "r": 8, "t": 36, "b": 8},
    "xaxis": {"gridcolor": "#1E2A3F", "zerolinecolor": "#2A3A55"},
    "yaxis": {"gridcolor": "#1E2A3F", "zerolinecolor": "#2A3A55"},
}


def inject_styles() -> None:
    """Suntik design system sekali per halaman."""
    st.markdown(_CSS, unsafe_allow_html=True)


def plotly_base(fig):
    """Template gelap konsisten untuk figure Plotly analitik."""
    fig.update_layout(
        font=_PLOTLY_DARK["font"],
        paper_bgcolor=_PLOTLY_DARK["paper_bgcolor"],
        plot_bgcolor=_PLOTLY_DARK["plot_bgcolor"],
        margin=_PLOTLY_DARK["margin"],
    )
    return fig


def chart(fig, key=None):
    """Render Plotly TANPA toolbar modebar."""
    kwargs = {"use_container_width": True, "config": {"displayModeBar": False}}
    if key is not None:
        kwargs["key"] = key
    st.plotly_chart(fig, **kwargs)


def hero() -> None:
    st.markdown(
        """
        <div class="fnd-hero">
          <div class="fnd-kicker">Detektor Hoaks · ID &amp; EN</div>
          <h1 class="fnd-title">Periksa berita sebelum kamu sebarkan.</h1>
          <p class="fnd-lede">Tempel tautan berita, model TF-IDF + Logistic Regression menilai
          keasliannya dan menunjukkan kata-kata yang memengaruhi vonis.</p>
          <ol class="fnd-steps">
            <li><span class="n">1</span>Tempel URL berita</li>
            <li><span class="n">2</span>Ekstrak &amp; deteksi</li>
            <li><span class="n">3</span>Baca vonis + bukti kata</li>
          </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, sub: str = "") -> None:
    st.markdown(
        f'<p class="fnd-section">{_html.escape(title)}</p>'
        + (f'<p class="fnd-sub">{_html.escape(sub)}</p>' if sub else ""),
        unsafe_allow_html=True,
    )


def verdict_banner(prediction: str, confidence: float) -> None:
    """Vonis besar pengganti st.error/st.success generik."""
    is_fake = str(prediction).upper() == "FAKE"
    cls = "fake" if is_fake else "real"
    name = "Terindikasi Hoaks" if is_fake else "Terindikasi Valid"
    raw = _html.escape(str(prediction))
    note = (
        "Skor keyakinan di bawah 70% — perlakukan sebagai sinyal awal, "
        "verifikasi ke sumber primer."
        if confidence < 0.7
        else "Vonis model, bukan kebenaran mutlak — tetap cek sumber primer."
    )
    st.markdown(
        f"""
        <div class="fnd-verdict {cls}" role="status">
          <div class="v-label">Hasil deteksi · {raw}</div>
          <div class="v-name">{name}</div>
          <div class="v-conf">Keyakinan {confidence:.1%}</div>
          <div class="v-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def confidence_meter(confidence: float) -> None:
    """Meter keyakinan HTML: angka besar + bilah + ambang 50/70."""
    pct = max(0.0, min(1.0, confidence)) * 100
    st.markdown(
        f"""
        <div class="fnd-meter" role="img" aria-label="Skor keyakinan {pct:.1f} persen">
          <div class="m-top">
            <span class="m-label">Skor keyakinan</span>
            <span class="m-value">{pct:.1f}%</span>
          </div>
          <div class="m-track">
            <div class="m-fill" style="width:{pct:.1f}%"></div>
            <div class="m-tick" style="left:50%" title="Ambang keputusan 50%"></div>
            <div class="m-tick" style="left:70%" title="Ambang andal 70%"></div>
          </div>
          <div class="m-scale"><span>0</span><span>50</span><span>70</span><span>100</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def probability_bars(probabilities: dict) -> None:
    """Dua baris probabilitas Hoaks vs Valid."""
    prob_dict = {k.upper(): float(v) for k, v in (probabilities or {}).items()}
    fake = prob_dict.get("FAKE", 0.0)
    real = prob_dict.get("REAL", 0.0)
    total = fake + real
    if total > 0:
        fake, real = fake / total, real / total
    st.markdown(
        f"""
        <div class="fnd-prob" role="img" aria-label="Hoaks {fake:.0%}, Valid {real:.0%}">
          <div class="p-row">
            <span class="p-name">Valid</span>
            <div class="p-track"><div class="p-fill real" style="width:{real * 100:.1f}%"></div></div>
            <span class="p-val">{real:.1%}</span>
          </div>
          <div class="p-row">
            <span class="p-name">Hoaks</span>
            <div class="p-track"><div class="p-fill fake" style="width:{fake * 100:.1f}%"></div></div>
            <span class="p-val">{fake:.1%}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prediction_badge(prediction: str) -> str:
    is_fake = str(prediction).upper() == "FAKE"
    cls = "badge-fake" if is_fake else "badge-real"
    label = "HOAKS" if is_fake else "VALID"
    return f'<span class="badge {cls}">{label}</span>'


def content_box(text: str, pre: bool = False) -> None:
    cls = "fnd-box pre" if pre else "fnd-box"
    st.markdown(
        f'<div class="{cls}">{_html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def empty_state(text: str) -> None:
    st.info(text)


def friendly_error(context: str, exc: Exception) -> None:
    """Error ramah untuk pengguna; detail teknis disembunyikan di expander."""
    st.error(f"{context}. Coba lagi, atau periksa URL dan koneksi.")
    with st.expander("Detail teknis"):
        st.code(f"{type(exc).__name__}: {exc}")
