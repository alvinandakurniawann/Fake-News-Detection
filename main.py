# main.py
"""
Aplikasi Streamlit News Scraper & Fake News Detector.
Presentasi via ui.py; alur ekstraksi, deteksi, dan riwayat tidak diubah.
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Import custom modules
from config import Config
from database_supabase import HistoryDatabase
from news_extractor import NewsExtractor
from text_preprocessor import TextPreprocessor
from tfidf_logreg_detector import TfidfLogregDetector
from ui import (
    RUST,
    TEAL,
    chart,
    confidence_meter,
    content_box,
    empty_state,
    friendly_error,
    hero,
    inject_styles,
    plotly_base,
    prediction_badge,
    probability_bars,
    section,
    verdict_banner,
)
from utils import process_batch_urls
from visualizations import highlight_important_words


@st.cache_resource(show_spinner=False)
def _get_extractor():
    """Satu instance dipakai ulang — jangan scrape-ulang setup tiap klik."""
    return NewsExtractor()


@st.cache_resource(show_spinner=False)
def _get_preprocessor():
    return TextPreprocessor()


@st.cache_resource(show_spinner="Memuat model...")
def _get_detector(model_path, model_type):
    """Model di-unpickle sekali saja. Tanpa ini, setiap interaksi
    memuat ulang model + koneksi DB hingga server kehabisan memori."""
    if model_type == "tfidf":
        return TfidfLogregDetector(model_path=model_path)
    return None


@st.cache_resource(show_spinner=False)
def _get_history_db(supabase_url, supabase_key):
    return HistoryDatabase(supabase_url=supabase_url, supabase_key=supabase_key)


def main():
    """Main application function"""
    st.set_page_config(
        page_title="Detektor Hoaks",
        page_icon="📰",
        layout="wide"
    )
    inject_styles()
    hero()

    # Validate Supabase configuration
    if not Config.validate_supabase_config():
        st.error("Konfigurasi Supabase tidak ditemukan.")
        st.info(
            "Aplikasi membutuhkan Supabase untuk menyimpan riwayat:\n"
            "1. Buat file `.streamlit/secrets.toml`\n"
            "2. Isi `SUPABASE_URL` dan `SUPABASE_KEY`."
        )
        st.stop()

    # Initialize session state for preprocessing steps if not exists
    if 'preprocessing_steps' not in st.session_state:
        st.session_state.preprocessing_steps = ['clean', 'punctuation', 'tokenize', 'stopwords', 'stem']

    # Initialize components (cached: dibuat sekali, dipakai ulang)
    extractor = _get_extractor()
    preprocessor = _get_preprocessor()
    st.session_state.preprocessor = preprocessor  # Store in session state

    # Get model configuration
    model_config = Config.get_model_config()

    # Initialize detector with the default model
    default_model = model_config["available_models"][0]

    # Initialize detector with the default model
    detector = _get_detector(default_model.get("path", ""), default_model.get("type", ""))
    if default_model.get("type") != "tfidf":
        st.error(f"Tipe model tidak didukung: {default_model.get('type')}")
        st.stop()
    if detector is None or not detector.model_loaded:
        st.error("Gagal menginisialisasi detector default.")
        st.stop()

    # Store detector and model config in session state
    st.session_state.detector = detector
    st.session_state.model_config = model_config
    st.session_state.current_model = default_model

    # Initialize Supabase database with config (cached client)
    config = Config.get_supabase_config()
    history_db = _get_history_db(config["supabase_url"], config["supabase_key"])

    # Initialize session state
    if 'extracted_data' not in st.session_state:
        st.session_state.extracted_data = None
    if 'preprocessed_text' not in st.session_state:
        st.session_state.preprocessed_text = None
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'explanation' not in st.session_state:
        st.session_state.explanation = None

    # Sidebar configuration
    setup_sidebar()
    preprocessing_steps = st.session_state.preprocessing_steps

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Pemeriksaan", "Batch", "Riwayat", "Analitik"])

    with tab1:
        single_check_tab(extractor, preprocessor, history_db, preprocessing_steps)

    with tab2:
        batch_processing_tab(extractor, preprocessor, history_db, preprocessing_steps)

    with tab3:
        history_tab(history_db)

    with tab4:
        analytics_tab(history_db)


def setup_sidebar():
    """Setup sidebar configuration"""
    # Initialize session state variables if they don't exist
    if 'preprocessing_steps' not in st.session_state:
        st.session_state.preprocessing_steps = Config.get_app_settings().get("preprocessing_defaults",
                                                                          ['clean', 'punctuation', 'tokenize', 'stopwords'])

    if 'current_model' not in st.session_state:
        st.session_state.current_model = {}

    with st.sidebar:
        st.header("Pengaturan")

        # Display model info - only TF-IDF model is available
        selected_model_info = st.session_state.model_config["available_models"][0]  # Get first (and only) model
        st.session_state.current_model = selected_model_info

        # Display model status
        if 'detector' in st.session_state and st.session_state.detector:
            model_info = st.session_state.detector.get_model_info()
            st.success(f"Model aktif: {model_info.get('name', 'TF-IDF + Logistic Regression')}")
            st.caption(f"Path: `{model_info.get('path', 'N/A')}`")
        else:
            st.warning("Model belum diinisialisasi")

        st.divider()

        # Preprocessing options
        st.subheader("Tahapan preprocessing")

        # Get available preprocessing steps from config or use defaults
        available_steps = Config.get_app_settings().get("available_preprocessing_steps",
                                                      ['clean', 'punctuation', 'tokenize', 'stopwords', 'stem'])

        # Update preprocessing steps in session state
        selected_steps = st.multiselect(
            "Langkah aktif:",
            available_steps,
            default=st.session_state.preprocessing_steps
        )

        # Update session state if selection changed
        if selected_steps != st.session_state.preprocessing_steps:
            st.session_state.preprocessing_steps = selected_steps

        with st.expander("Arti tiap tahapan"):
            st.markdown(
                "- **clean**: lowercase, hapus URL dan email\n"
                "- **punctuation**: hapus tanda baca\n"
                "- **tokenize**: pecah teks menjadi kata\n"
                "- **stopwords**: buang kata umum\n"
                "- **stem**: kembalikan ke kata dasar"
            )


def single_check_tab(extractor, preprocessor, history_db, preprocessing_steps):
    """Single URL checking tab"""
    col1, col2 = st.columns([1, 1])

    with col1:
        section("Tautan berita", "Hasil ekstraksi tampil di kanan setelah diproses.")

        # Check if URL exists in history
        url_input = st.text_input(
            "URL berita:",
            placeholder="https://www.detik.com/..."
        )

        # Supported domains hint
        domains = Config.get_app_settings().get("supported_domains", [])
        if domains:
            st.markdown(
                "<ul class=\"domains\">" + "".join(f"<li>{d}</li>" for d in domains) + "</ul>",
                unsafe_allow_html=True,
            )

        # Check history
        if url_input:
            existing_record = history_db.check_url_exists(url_input)
            if existing_record:
                st.info(f"URL ini pernah diperiksa pada {existing_record['checked_at']}")

        # Extract and predict buttons
        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            if st.button("Ekstrak saja", type="secondary", use_container_width=True):
                extract_only(url_input, extractor)

        with col_btn2:
            if st.button("Ekstrak & deteksi", type="primary", use_container_width=True):
                # Ambil detector dari session_state
                detector = st.session_state.get('detector')
                if detector:
                    extract_and_detect(url_input, extractor, preprocessor, detector,
                                     history_db, preprocessing_steps)
                else:
                    st.error("Detector belum diinisialisasi.")

    with col2:
        display_results()

    # Feature explanation section
    if st.session_state.explanation and st.session_state.extracted_data:
        display_explanation()


def extract_only(url_input, extractor):
    """Extract news without detection"""
    if url_input:
        with st.spinner("Mengekstrak berita..."):
            result = extractor.extract_from_url(url_input)

            if result['success']:
                st.session_state.extracted_data = result
                st.success("Ekstraksi berhasil.")
            else:
                st.error(f"Gagal mengekstrak: {result['error']}")
    else:
        st.warning("Isi URL terlebih dahulu.")


def extract_and_detect(url_input, extractor, preprocessor, detector, history_db, preprocessing_steps):
    """Extract news and detect fake news"""
    if url_input:
        with st.spinner("Mengekstrak dan menganalisis..."):
            # Extract
            result = extractor.extract_from_url(url_input)
            if result['success']:
                st.session_state.extracted_data = result
                # Preprocess
                full_text = f"{result['title']} {result['content']}"
                processed_text = preprocessor.preprocess_pipeline(full_text, preprocessing_steps)
                st.session_state.preprocessed_text = processed_text
                # Gunakan objek detector dari parameter
                if detector.model_loaded:
                    prediction = detector.predict(processed_text)
                    st.session_state.prediction_result = prediction
                    explanation = detector.explain_prediction(processed_text)
                    st.session_state.explanation = explanation

                    # Normalisasi format prediksi
                    pred = st.session_state.prediction_result
                    probs = pred['probabilities']

                    # Handle perbedaan format (FAKE/REAL vs fake/real)
                    if 'FAKE' in probs and 'REAL' in probs:
                        fake_prob = probs['FAKE']
                        real_prob = probs['REAL']
                    elif 'fake' in probs and 'real' in probs:
                        fake_prob = probs['fake']
                        real_prob = probs['real']
                    else:
                        # Fallback jika format tidak dikenali
                        fake_prob = 0.5
                        real_prob = 0.5

                    # Pastikan prediction dalam format yang konsisten
                    prediction_label = pred['prediction'].upper()

                    # Save to history
                    history_record = {
                        'url': url_input,
                        'domain': result['domain'],
                        'title': result['title'],
                        'content': result['content'],
                        'prediction': prediction_label,
                        'confidence': pred['confidence'],
                        'fake_probability': fake_prob,
                        'real_probability': real_prob,
                        'checked_at': datetime.now()
                    }
                    history_db.add_record(history_record)
                    st.success("Analisis selesai.")
                else:
                    st.error("Model detector belum dimuat.")
            else:
                st.error(f"Gagal mengekstrak: {result['error']}")
    else:
        st.warning("Isi URL terlebih dahulu.")


def display_results():
    """Display extraction and prediction results"""
    if 'extracted_data' not in st.session_state or not st.session_state.extracted_data:
        empty_state("Belum ada data. Tempel URL di kiri lalu jalankan ekstraksi.")
        return

    data = st.session_state.extracted_data

    # Check if data is valid
    if not isinstance(data, dict) or 'title' not in data or 'content' not in data:
        st.error("Format data tidak valid, hasil tidak bisa ditampilkan.")
        return

    try:
        # Display basic info
        section("Berita terekstrak")
        st.markdown(
            f"<p class=\"fnd-meta\"><b>{data.get('title', 'Tanpa judul')}</b><br>"
            f"Sumber: {data.get('domain', 'Tidak diketahui')} · "
            f"Tanggal: {data.get('publish_date', 'N/A')}</p>",
            unsafe_allow_html=True,
        )
        if data.get('url'):
            st.markdown(f"[Buka artikel asli]({data['url']})")

        # Add export buttons
        col1, col2 = st.columns(2)

        with col1:
            # CSV Export
            csv = pd.DataFrame([{
                'title': data.get('title', ''),
                'source': data.get('domain', ''),
                'publish_date': data.get('publish_date', ''),
                'content': data.get('content', ''),
                'url': data.get('url', '')
            }]).to_csv(index=False)
            st.download_button(
                label="Unduh CSV",
                data=csv,
                file_name=f"news_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col2:
            # JSON Export
            json_data = {
                'title': data.get('title', ''),
                'source': data.get('domain', ''),
                'publish_date': data.get('publish_date', ''),
                'content': data.get('content', ''),
                'url': data.get('url', '')
            }
            st.download_button(
                label="Unduh JSON",
                data=json.dumps(json_data, indent=2, ensure_ascii=False),
                file_name=f"news_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )

        st.divider()

        # Create tabs for original and preprocessed content
        tab1, tab2 = st.tabs(["Teks asli", "Teks preprocess"])

        with tab1:
            if data.get('content'):
                content_box(data['content'])
            else:
                st.warning("Konten tidak tersedia.")

        with tab2:
            if data.get('content'):
                # Get preprocessed text from session state if available
                preprocessed_text = st.session_state.get('preprocessed_text')

                if preprocessed_text:
                    content_box(preprocessed_text, pre=True)
                    st.caption(
                        f"Telah diproses dengan: {', '.join(st.session_state.get('preprocessing_steps', []))}"
                    )

                    # Add copy button for preprocessed text
                    st.download_button(
                        label="Salin teks preprocess",
                        data=preprocessed_text,
                        file_name="preprocessed_text.txt",
                        mime="text/plain"
                    )
                else:
                    st.info("Teks preprocess belum ada. Jalankan deteksi untuk melihatnya.")
            else:
                st.warning("Konten tidak tersedia untuk diproses.")

        # Display prediction if available
        if 'prediction_result' in st.session_state and st.session_state.prediction_result:
            prediction = st.session_state.prediction_result

            section("Hasil prediksi")

            # Validate prediction data
            if isinstance(prediction, dict) and 'prediction' in prediction and 'confidence' in prediction:
                verdict_banner(prediction['prediction'], prediction['confidence'])

                # Meter keyakinan + distribusi probabilitas (HTML, tanpa toolbar)
                if 'confidence' in prediction and prediction['confidence'] is not None:
                    confidence_meter(prediction['confidence'])

                if 'probabilities' in prediction and prediction['probabilities'] is not None:
                    probability_bars(prediction['probabilities'])
            else:
                st.warning("Data prediksi tidak lengkap.")
    except Exception as e:
        friendly_error("Gagal menampilkan hasil", e)


def display_explanation():
    """Display feature explanation"""
    st.divider()
    section("Bukti kata", "Kata yang memengaruhi vonis model. Arahkan kursor ke kata untuk melihat bobotnya.")

    # Get preprocessing steps from session state or use default
    preprocessing_steps = st.session_state.get('preprocessing_steps',
        ['clean', 'normalize', 'remove_stopwords', 'stem'])

    col3, col4 = st.columns([2, 1])

    with col3:
        section("Teks tersorot")
        full_text = f"{st.session_state.extracted_data['title']} {st.session_state.extracted_data['content']}"

        # Get preprocessor from session state
        preprocessor = st.session_state.get('preprocessor')

        # Show first 2000 characters by default, with option to show more
        show_full_text = st.checkbox("Tampilkan teks penuh", value=False, key="show_full_text")

        if show_full_text:
            text_to_show = full_text
            show_less = "(Menampilkan teks penuh)"
        else:
            text_to_show = full_text[:2000] + ("..." if len(full_text) > 2000 else "")
            show_less = ""

        # Highlight important words
        highlighted_html = highlight_important_words(
            text_to_show,
            st.session_state.explanation['important_words'],
            preprocessor=preprocessor,
            preprocessing_steps=preprocessing_steps
        )

        # Display the text with highlighting (theme-aware box)
        st.markdown(
            f'<div class="fnd-box">{highlighted_html}</div>'
            f'<p class="fnd-meta">{show_less}</p>',
            unsafe_allow_html=True
        )

        # Show word count info
        word_count = len(full_text.split())
        st.caption(f"Total {word_count} kata · tampil: {'semua' if show_full_text or len(full_text) <= 2000 else '2000 karakter pertama'}")

    with col4:
        section("Bobot kata")
        words = st.session_state.explanation['important_words']
        if words:
            words_df = pd.DataFrame(words)
            if 'weight' in words_df.columns:
                words_df['weight'] = words_df['weight'].round(3)
                # Sort by absolute weight for better visualization
                words_df = words_df.iloc[words_df['weight'].abs().argsort()[::-1]]
            st.dataframe(words_df, hide_index=True, use_container_width=True)

            # Add color legend
            st.markdown(
                """<div class="hl-legend">
                <span><i style="background:rgba(249,115,22,0.5)"></i>ke hoaks</span>
                <span><i style="background:rgba(34,211,238,0.5)"></i>ke valid</span>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.info("Penjelasan fitur tidak tersedia untuk model ini.")


def batch_processing_tab(extractor, preprocessor, history_db, preprocessing_steps):
    """Batch processing tab"""
    section("Proses banyak URL", "Satu URL per baris. Maksimal 20 URL per batch.")

    urls_input = st.text_area(
        "Daftar URL:",
        height=150,
        placeholder="https://www.detik.com/...\nhttps://www.kompas.com/..."
    )

    if st.button("Proses batch", type="primary"):
        if urls_input:
            urls = [url.strip() for url in urls_input.split('\n') if url.strip()]

            if urls:
                # Ambil detector dari session_state
                detector = st.session_state.get('detector')
                if detector and detector.model_loaded:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    with st.spinner(f"Memproses {len(urls)} URL..."):
                        # Process batch menggunakan detector dari session_state
                        results = process_batch_urls(
                            urls, extractor, preprocessor, detector, preprocessing_steps
                        )

                        # Update progress
                        progress_bar.progress(1.0)
                        status_text.text(f"Selesai memproses {len(results)} URL")

                    # Display results
                    display_batch_results(results, history_db)
                elif detector:
                    st.error("Model detector belum dimuat.")
                else:
                    st.error("Detector belum diinisialisasi.")
        else:
            st.warning("Isi minimal satu URL.")


def display_batch_results(results, history_db):
    """Display batch processing results"""
    section("Hasil batch")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    success_count = sum(1 for r in results if r.get('status') == 'success')
    fake_count = sum(1 for r in results if r.get('prediction') == 'FAKE')
    real_count = sum(1 for r in results if r.get('prediction') == 'REAL')
    avg_confidence = np.mean([r.get('confidence', 0) for r in results if r.get('confidence')])

    with col1:
        st.metric("Total URL", len(results))
    with col2:
        st.metric("Berhasil", success_count)
    with col3:
        st.metric("Hoaks", fake_count)
    with col4:
        st.metric("Rata-rata keyakinan", f"{avg_confidence:.0%}")

    # Display detailed results for each URL
    for idx, result in enumerate(results, 1):
        with st.expander(f"{idx}. {result.get('title', 'Tanpa judul')}", expanded=False):
            col1, col2 = st.columns([1, 1])

            with col1:
                # Display URL and domain
                st.markdown(f"[Buka artikel]({result.get('url', '')}) · {result.get('domain', 'N/A')}")

                # Display prediction and confidence
                if result.get('status') == 'success':
                    prediction = result.get('prediction', 'UNKNOWN')
                    confidence = result.get('confidence', 0)

                    # Prediction badge
                    st.markdown(
                        f"{prediction_badge(prediction)} "
                        f"<span class=\"mono\" style=\"font-size:0.85rem;\">{confidence:.0%}</span>",
                        unsafe_allow_html=True,
                    )

                    # Meter keyakinan (HTML, tanpa toolbar)
                    confidence_meter(confidence)

                    # Distribusi probabilitas (HTML, tanpa toolbar)
                    probability_bars({
                        'FAKE': result.get('fake_probability', 0),
                        'REAL': result.get('real_probability', 0)
                    })
                else:
                    st.error(f"Gagal memproses: {result.get('error', 'Error tidak diketahui')}")

            with col2:
                if result.get('status') == 'success' and 'content' in result:
                    # Display important words if available
                    if 'important_words' in result:
                        section("Kata berpengaruh")
                        important_words = result['important_words']

                        # Display word importance table
                        words_df = pd.DataFrame(important_words)
                        words_df['weight'] = words_df['weight'].round(4)
                        st.dataframe(
                            words_df,
                            column_config={
                                "word": "Word",
                                "weight": st.column_config.NumberColumn(
                                    "Importance",
                                    format="%.4f"
                                )
                            },
                            hide_index=True,
                            use_container_width=True
                        )

                        # Combine title and content for highlighting
                        full_text = f"{result.get('title', '')} {result.get('content', '')}"

                        # Add 'Show more' functionality
                        show_full_text = st.checkbox(
                            "Tampilkan teks penuh",
                            value=False,
                            key=f"show_full_{result['url']}"
                        )

                        if show_full_text:
                            text_to_show = full_text
                            show_less = "(Menampilkan teks penuh)"
                        else:
                            text_to_show = full_text[:2000] + ("..." if len(full_text) > 2000 else "")
                            show_less = ""

                        # Display highlighted text
                        section("Teks tersorot")

                        # Get preprocessor from session state
                        preprocessor = st.session_state.get('preprocessor')
                        preprocessing_steps = st.session_state.get('preprocessing_steps', [])

                        # Highlight important words
                        highlighted = highlight_important_words(
                            text_to_show,
                            important_words,
                            preprocessor=preprocessor,
                            preprocessing_steps=preprocessing_steps
                        )

                        # Display the text with highlighting
                        st.markdown(
                            f'<div class="fnd-box">{highlighted}</div>'
                            f'<p class="fnd-meta">{show_less}</p>',
                            unsafe_allow_html=True
                        )

                        # Show word count info
                        word_count = len(full_text.split())
                        st.caption(f"Total {word_count} kata")

    # Save successful results to history
    results_df = pd.DataFrame([r for r in results if r.get('status') == 'success'])
    if not results_df.empty:
        section("Tabel ringkasan")
        st.dataframe(results_df[['url', 'domain', 'prediction', 'confidence']], hide_index=True)

        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Unduh hasil CSV",
            data=csv,
            file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

        # Save to history database
        for _, result in results_df.iterrows():
            history_record = {
                'url': result['url'],
                'domain': result['domain'],
                'title': result['title'],
                'content': result.get('content', ''),
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'fake_probability': result['fake_probability'],
                'real_probability': result['real_probability'],
                'checked_at': datetime.now()
            }
            history_db.add_record(history_record)


def history_tab(history_db):
    """History tab"""
    section("Riwayat pemeriksaan", "Seratus pemeriksaan terakhir, bisa disaring dan dicari.")

    # Get history
    history_df = history_db.get_history(limit=100)

    if not history_df.empty:
        display_history(history_df)
    else:
        empty_state("Belum ada riwayat. Periksa beberapa URL terlebih dahulu.")


def display_history(history_df):
    """Display history with filters"""
    # History metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total", len(history_df))
    with col2:
        fake_pct = (history_df['prediction'] == 'FAKE').sum() / len(history_df) * 100
        st.metric("Hoaks", f"{fake_pct:.0f}%")
    with col3:
        avg_conf = history_df['confidence'].mean()
        st.metric("Rata-rata keyakinan", f"{avg_conf:.0%}")
    with col4:
        unique_domains = history_df['domain'].nunique()
        st.metric("Domain unik", unique_domains)

    # Filter options
    col5, col6 = st.columns([1, 3])

    with col5:
        filter_prediction = st.selectbox(
            "Saring vonis:",
            ["Semua", "FAKE", "REAL"]
        )

    with col6:
        search_term = st.text_input("Cari judul:", "")

    # Apply filters
    filtered_df = history_df.copy()

    if filter_prediction != "Semua":
        filtered_df = filtered_df[filtered_df['prediction'] == filter_prediction]

    if search_term:
        filtered_df = filtered_df[
            filtered_df['title'].str.contains(search_term, case=False, na=False)
        ]

    # Display filtered history
    if filtered_df.empty:
        empty_state("Tidak ada baris yang cocok dengan saringan.")
    else:
        st.dataframe(filtered_df, hide_index=True, use_container_width=True)

    # Export history
    if st.button("Ekspor riwayat penuh"):
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="Unduh CSV riwayat",
            data=csv,
            file_name=f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


def analytics_tab(history_db):
    """Analytics tab"""
    section("Analitik", "Agregat dari seribu pemeriksaan terakhir.")

    history_df = history_db.get_history(limit=1000)

    if not history_df.empty:
        display_analytics(history_df)
    else:
        empty_state("Belum ada data analitik. Periksa beberapa URL terlebih dahulu.")


def display_analytics(history_df):
    """Display analytics charts"""
    # Convert checked_at to datetime
    history_df['checked_at'] = pd.to_datetime(history_df['checked_at'])

    # Prediction distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sebaran vonis")
        pred_counts = history_df['prediction'].value_counts()
        fig_pie = px.pie(
            values=pred_counts.values,
            names=pred_counts.index,
            color=pred_counts.index,
            color_discrete_map={'FAKE': RUST, 'REAL': TEAL},
        )
        fig_pie.update_traces(textinfo="percent+label")
        chart(plotly_base(fig_pie))

    with col2:
        st.subheader("Sebaran keyakinan")
        fig_hist = px.histogram(
            history_df,
            x='confidence',
            nbins=20,
            color_discrete_sequence=[TEAL],
        )
        fig_hist.update_xaxes(title='Skor keyakinan', tickformat=".0%")
        fig_hist.update_yaxes(title='Jumlah')
        chart(plotly_base(fig_hist))

    # Domain analysis
    st.subheader("Domain tersering diperiksa")
    domain_counts = history_df['domain'].value_counts().head(10)
    fig_domains = px.bar(
        x=domain_counts.values,
        y=domain_counts.index,
        orientation='h',
        labels={'x': 'Jumlah pemeriksaan', 'y': 'Domain'},
        color_discrete_sequence=[TEAL],
    )
    fig_domains.update_yaxes(categoryorder="total ascending")
    chart(plotly_base(fig_domains))

    # Fake news by domain
    st.subheader("Laju hoaks per domain")
    domain_fake_rate = history_df.groupby('domain').agg({
        'prediction': lambda x: (x == 'FAKE').sum() / len(x) * 100
    }).round(1)
    domain_fake_rate = domain_fake_rate.sort_values('prediction', ascending=True).head(10)

    fig_fake_rate = px.bar(
        x=domain_fake_rate['prediction'],
        y=domain_fake_rate.index,
        orientation='h',
        labels={'x': 'Laju hoaks (%)', 'y': 'Domain'},
        color_discrete_sequence=[RUST],
    )
    chart(plotly_base(fig_fake_rate))
    st.caption("Hanya 10 domain teratas. Domain dengan 1 pemeriksaan bisa menunjukkan 0% atau 100%.")


if __name__ == "__main__":
    main()
