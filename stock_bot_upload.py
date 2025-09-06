import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from urllib.parse import urlparse
import html

# -----------------------------
# Config & API keys (free)
# -----------------------------
st.set_page_config(page_title="📊 Stock Analysis Dashboard", layout="wide")

# Replace with your actual API keys (free tiers)
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", "demo")
MARKETAUX_API_KEY = st.secrets.get("MARKETAUX_API_KEY", "demo")

# --- News section styles ---
st.markdown("""
<style>
.news-item { margin: 0.6rem 0 1.0rem 0; }

/* Headline row used as a toggle */
.news-details { margin-left: 0; }
.news-details summary { cursor: pointer; list-style: none; }
.news-details summary::-webkit-details-marker { display: none; }
.news-details summary::before { content: "▸ "; color: #6b7280; }
.news-details[open] summary::before { content: "▾ "; }

/* Headline row layout */
.news-headline { 
    font-size: 1.05rem; 
    font-weight: 600; 
    line-height: 1.35;
    display: inline;
}

/* Sentiment emoji spacing (after datetime) */
.sentiment-emoji {
    margin-left: 6px;
    display: inline-block;
}

/* Date next to headline */
.news-bracket { color: #6b7280; font-size: 0.92rem; margin-left: 6px; }
            
/* Bullet points for expanded content (source and sentiment only) */
.news-body ul {
    margin: 0.35rem 0 0 0;
    padding-left: 1.25rem;
    list-style-type: none;
}

.news-body li {
    margin-bottom: 0.25rem;
    position: relative;
    font-size: 0.92rem;
    color: #374151;
}

.news-body li::before {
    content: "•";
    color: #6b7280;
    font-weight: bold;
    display: inline-block;
    width: 1em;
    margin-left: -1em;
    position: absolute;
    left: 0;
}

/* Summary without bullet */
.news-summary {
    font-size: 0.92rem;
    color: #374151;
    margin: 0.35rem 0 0.5rem 0;
    padding-left: 0.5rem;
}

/* Remove the bullet from summary */
.news-summary::before {
    content: none;
}

/* Source line + link inside expanded area */
.news-meta { font-size: 0.92rem; color: #374151; margin-bottom: 0.25rem; }
.news-domain { color: #2563eb; text-decoration: none; }
.news-domain:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Helpers
# -----------------------------

def extract_domain(url: str | None) -> str:
    """Return 'example.com' from a URL (no scheme or path)."""
    if not url:
        return ""
    try:
        netloc = urlparse(url).netloc or ""
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return ""

def normalize_url(url):
    """Return a normalized URL with scheme, or None."""
    if not url or not isinstance(url, str):
        return None
    u = url.strip()
    if not u:
        return None
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    return u

def safe_dt_str(dt_str):
    """Format a date string safely as 'YYYY-MM-DD HH:MM' or return raw string."""
    if not dt_str:
        return ""
    try:
        return pd.to_datetime(dt_str).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(dt_str)

# -----------------------------
# Data Fetching (cached)
# -----------------------------
@st.cache_data(ttl=300)  # cache for 5 minutes
def fetch_stock_data(symbol):
    url = (
        "https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY&symbol={symbol}"
        f"&apikey={ALPHA_VANTAGE_API_KEY}&outputsize=compact"
    )
    try:
        resp = requests.get(url, timeout=30)
        data = resp.json()
    except Exception:
        return None

    if "Time Series (Daily)" not in data:
        return None

    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index", dtype=float)
    # Expected columns: '1. open','2. high','3. low','4. close','5. volume'
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

# -----------------------------
# Indicators
# -----------------------------
def calculate_rsi(df, period=14):
    close_series = df['4. close']
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(df):
    close_series = df['4. close']
    ema12 = close_series.ewm(span=12, adjust=False).mean()
    ema26 = close_series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def detect_macd_signals(macd, signal):
    macd = macd.dropna()
    signal = signal.reindex(macd.index).dropna()
    macd = macd.reindex(signal.index)
    buy_signals, sell_signals = [], []
    for i in range(1, len(macd)):
        pm, ps = macd.iloc[i-1], signal.iloc[i-1]
        cm, cs = macd.iloc[i], signal.iloc[i]
        if pm < ps and cm > cs:
            buy_signals.append((macd.index[i], cm))
        elif pm > ps and cm < cs:
            sell_signals.append((macd.index[i], cm))
    return buy_signals, sell_signals

def sma(series, window):
    return series.rolling(window).mean()

def bollinger_bands(close_series, window=20, mult=2.0):
    mid = close_series.rolling(window).mean()
    std = close_series.rolling(window).std()
    upper = mid + mult * std
    lower = mid - mult * std
    return upper, mid, lower

def stochastic_oscillator(df, k_period=14, d_period=3):
    high = df['2. high']
    low = df['3. low']
    close = df['4. close']
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    return k, d

def atr(df, period=14):
    high = df['2. high']
    low = df['3. low']
    close = df['4. close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def sentiment_to_emoji(sentiment_score):
    """Convert sentiment score to an appropriate emoji"""
    if sentiment_score is None:
        return "❓"  # Unknown sentiment
    elif sentiment_score > 0.5:
        return "😊"  # Positive
    elif sentiment_score < 0.3:
        return "😟"  # Negative
    else:
        return "😐"  # Neutral

def sentiment_to_text(sentiment_score):
    """Convert sentiment score to descriptive text"""
    if sentiment_score is None:
        return "Unknown sentiment"
    elif sentiment_score > 0.5:
        return f"Positive ({sentiment_score:.2f})"
    elif sentiment_score < 0.3:
        return f"Negative ({sentiment_score:.2f})"
    else:
        return f"Neutral ({sentiment_score:.2f})"

# -----------------------------
# News & sentiment (cached)
# -----------------------------
@st.cache_data(ttl=300)
def fetch_news_sentiment(symbol):
    """
    Returns (avg_sentiment: float|None, articles: list[dict])
    Each article: {title, url, source, published_at, sentiment, description}
    """
    url = (
        f"https://api.marketaux.com/v1/news/all"
        f"?symbols={symbol}&filter_entities=true&language=en&api_token={MARKETAUX_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=30)
        data = resp.json()
    except Exception:
        return None, []

    if "data" not in data:
        return None, []

    sentiments = []
    articles = []
    for article in data["data"]:
        overall = article.get("overall_sentiment_score")
        ent_score = None
        if article.get("entities"):
            ent0 = article["entities"][0]
            ent_score = ent0.get("sentiment_score")
        score = overall if overall is not None else ent_score
        if score is not None:
            sentiments.append(score)

        desc = article.get("description") or article.get("snippet") or article.get("summary")
        source_name = (
            (article.get("source") or {}).get("name")
            if isinstance(article.get("source"), dict)
            else article.get("source")
        ) or "Unknown Source"

        articles.append({
            "title": article.get("title", "No title"),
            "url": normalize_url(article.get("url")),
            "source": source_name,
            "published_at": article.get("published_at"),
            "sentiment": score,
            "description": desc
        })

    avg_sentiment = (sum(sentiments) / len(sentiments)) if sentiments else None

    try:
        articles = sorted(
            articles,
            key=lambda a: pd.to_datetime(a.get("published_at")) if a.get("published_at") else pd.Timestamp.min,
            reverse=True
        )[:5]
    except Exception:
        articles = articles[:5]

    return avg_sentiment, articles

# -----------------------------
# Recommendation logic
# -----------------------------
def combined_recommendation(rsi_val, macd_signal, sentiment):
    score = 0
    rsi_text = "Neutral"
    macd_text = "Neutral"
    sentiment_text = "Neutral"

    if rsi_val < 30:
        score += 1
        rsi_text = "📉 RSI < 30: Oversold (Buy)"
    elif rsi_val > 70:
        score -= 1
        rsi_text = "📈 RSI > 70: Overbought (Sell)"
    else:
        rsi_text = "📊 RSI in neutral range"

    if macd_signal == "buy":
        score += 1
        macd_text = "🟢 MACD crossover: Buy signal"
    elif macd_signal == "sell":
        score -= 1
        macd_text = "🔴 MACD crossover: Sell signal"
    else:
        macd_text = "⚪ No recent MACD crossover"

    if sentiment is not None:
        sentiment_emoji = sentiment_to_emoji(sentiment)
        sentiment_description = sentiment_to_text(sentiment)
        sentiment_text = f"{sentiment_emoji} {sentiment_description}"
        
        if sentiment > 0.5:
            score += 1
        elif sentiment < 0.3:
            score -= 1
    else:
        sentiment_text = "❓ No sentiment data"

    if score >= 2:
        return "✅ Strong Buy Recommendation", rsi_text, macd_text, sentiment_text, "success"
    elif score == 1:
        return "☑️ Moderate Buy Recommendation", rsi_text, macd_text, sentiment_text, "info"
    elif score == 0:
        return "ℹ️ Hold Recommendation", rsi_text, macd_text, sentiment_text, "warning"
    else:
        return "⚠️ Sell Recommendation", rsi_text, macd_text, sentiment_text, "error"

# -----------------------------
# UI
# -----------------------------
st.title("📊 Stock Analysis Dashboard")
symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT)").strip().upper()

if symbol:
    df = fetch_stock_data(symbol)
    if df is None or df.empty:
        st.error("Failed to fetch stock data. Please check the symbol and try again.")
        st.stop()

    # Window for display (no date selector needed)
    LOOKBACK_DAYS = 100
    idx_plot = df.index[-LOOKBACK_DAYS:]
    df_plot = df.loc[idx_plot]

    # Define series early (prevents NameError)
    open_series   = df['1. open']
    high_series   = df['2. high']
    low_series    = df['3. low']
    close_series  = df['4. close']
    volume_series = df['5. volume']

    latest_update_date = df.index[-1].strftime("%Y-%m-%d")
    st.caption(f"📅 Latest stock data update: {latest_update_date}")

    # Compute indicators
    rsi = calculate_rsi(df)
    macd, signal, hist = calculate_macd(df)
    buy_signals, sell_signals = detect_macd_signals(macd, signal)
    stoch_k, stoch_d = stochastic_oscillator(df)
    atr14 = atr(df)

    # Precompute overlays on full history, then slice to window
    sma20_full = sma(close_series, 20)
    sma50_full = sma(close_series, 50)
    bb_up_series, bb_mid_series, bb_low_series = bollinger_bands(close_series, window=20, mult=2.0)

    sma20 = sma20_full.reindex(idx_plot)
    sma50 = sma50_full.reindex(idx_plot)
    bb_up = bb_up_series.reindex(idx_plot)
    bb_low = bb_low_series.reindex(idx_plot)

    # Latest values
    latest_close = close_series.iloc[-1]
    latest_rsi = float(rsi.iloc[-1])
    latest_stoch_k = float(stoch_k.iloc[-1]) if not np.isnan(stoch_k.iloc[-1]) else None
    latest_stoch_d = float(stoch_d.iloc[-1]) if not np.isnan(stoch_d.iloc[-1]) else None
    latest_atr = float(atr14.iloc[-1]) if not np.isnan(atr14.iloc[-1]) else None

    # Last-bar MACD signal
    latest_macd_signal = "none"
    if len(macd) >= 2 and len(signal) >= 2:
        if macd.iloc[-2] < signal.iloc[-2] and macd.iloc[-1] > signal.iloc[-1]:
            latest_macd_signal = "buy"
        elif macd.iloc[-2] > signal.iloc[-2] and macd.iloc[-1] < signal.iloc[-1]:
            latest_macd_signal = "sell"

    sentiment_score, articles = fetch_news_sentiment(symbol)

    # Recommendation banner
    recommendation, rsi_text, macd_text, sentiment_text, banner_type = combined_recommendation(
        latest_rsi, latest_macd_signal, sentiment_score
    )
    getattr(st, banner_type)(recommendation)

    # KPI row
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric("Last Close", f"${latest_close:,.2f}")
    with kpi2:
        st.metric("RSI (14)", f"{latest_rsi:.2f}")
    with kpi3:
        st.metric("Avg Sentiment", f"{sentiment_score:.2f}" if sentiment_score is not None else "N/A")

    # Row: Indicators (left) and News (right)
    col_ind, col_news = st.columns([1, 1], gap="large")

    with col_ind:
        st.subheader("🎯 Indicators")
        # Indicators text (includes RSI figure)
        st.markdown(
            f"- {rsi_text} **(RSI: {latest_rsi:.2f})**\n"
            f"- {macd_text}\n"
            f"- {sentiment_text}"
        )

        # Indicator snapshot (compact)
        sma_bias = (
            "Bullish (SMA20 > SMA50)" if (sma20_full.iloc[-1] > sma50_full.iloc[-1]) else
            "Bearish (SMA20 < SMA50)" if (sma20_full.iloc[-1] < sma50_full.iloc[-1]) else
            "Neutral"
        )
        stoch_text = (
            f"%K: {latest_stoch_k:.1f}, %D: {latest_stoch_d:.1f} — "
            + ("Overbought (>80)" if (latest_stoch_k is not None and latest_stoch_k > 80) else
               "Oversold (<20)" if (latest_stoch_k is not None and latest_stoch_k < 20) else
               "Neutral")
        ) if (latest_stoch_k is not None and latest_stoch_d is not None) else "N/A"
        atr_text = f"{latest_atr:.2f}" if latest_atr is not None else "N/A"
        st.markdown(
            f"- **SMA Bias:** {sma_bias}\n"
            f"- **Stochastic (14,3):** {stoch_text}\n"
            f"- **ATR (14):** {atr_text}"
        )

    with col_news:
        st.subheader("📰 Recent News")

        if articles:
            for a in articles:
                title = a.get("title", "No title") or "No title"
                raw_url = a.get("url")
                url = normalize_url(raw_url)
                ts = safe_dt_str(a.get("published_at"))
                domain = extract_domain(url) if url else (a.get("source") or "")
                sentiment = a.get("sentiment")
                sentiment_emoji = sentiment_to_emoji(sentiment)

                # Headline first, then datetime, then sentiment emoji
                headline_html = html.escape(title)
                bracket_html = f'<span class="news-bracket">({html.escape(ts)})</span>' if ts else ""
                sentiment_html = f'<span class="sentiment-emoji">{sentiment_emoji}</span>' if sentiment_emoji else ""

                # Expanded content
                desc = a.get("description") or a.get("snippet") or a.get("summary") or ""

                # Build source with bullet point
                source_bullet = ""
                if url and domain:
                    source_bullet = f'<li>Source: <a href="{html.escape(url)}" target="_blank" class="news-domain">{html.escape(domain)}</a></li>'
                elif domain:
                    source_bullet = f'<li>Source: {html.escape(domain)}</li>'

                # Build sentiment with bullet point
                sentiment_bullet = ""
                if sentiment is not None:
                    sentiment_text = sentiment_to_text(sentiment)
                    sentiment_bullet = f'<li>Sentiment: {sentiment_text}</li>'

                # Build the content
                content_html = ""
                if desc:
                    content_html += f'<div class="news-summary">{html.escape(desc)}</div>'
                
                # Add bullet points for source and sentiment
                if source_bullet or sentiment_bullet:
                    content_html += '<ul>'
                    if source_bullet:
                        content_html += source_bullet
                    if sentiment_bullet:
                        content_html += sentiment_bullet
                    content_html += '</ul>'

                st.markdown(
                    f"""
                    <div class="news-item">
                    <details class="news-details">
                        <summary>
                        <span class="news-headline">{headline_html}</span>
                        {bracket_html}
                        {sentiment_html}
                        </summary>
                        <div class="news-body">
                        {content_html}
                        </div>
                    </details>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No news articles available at the moment.")

    st.subheader("📈 Trends")

    # Price + Volume (top)
    fig_price = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.72, 0.28],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )

    # Candles
    fig_price.add_trace(
        go.Candlestick(
            x=idx_plot,
            open=df_plot['1. open'],
            high=df_plot['2. high'],
            low=df_plot['3. low'],
            close=df_plot['4. close'],
            name="Price",
            increasing=dict(line=dict(color="#00B26F")),
            decreasing=dict(line=dict(color="#E45756")),
        ),
        row=1, col=1
    )

    # SMA 20/50
    fig_price.add_trace(go.Scatter(x=idx_plot, y=sma20, mode='lines', name='SMA 20',
                                   line=dict(color="#1f77b4", width=1.5)), row=1, col=1)
    fig_price.add_trace(go.Scatter(x=idx_plot, y=sma50, mode='lines', name='SMA 50',
                                   line=dict(color="#ff7f0e", width=1.5)), row=1, col=1)

    # Bollinger bands (upper/lower)
    fig_price.add_trace(go.Scatter(x=idx_plot, y=bb_up, mode='lines', name='BB Upper',
                                   line=dict(color="rgba(31,119,180,0.4)", width=1)), row=1, col=1)
    fig_price.add_trace(go.Scatter(x=idx_plot, y=bb_low, mode='lines', name='BB Lower',
                                   line=dict(color="rgba(31,119,180,0.4)", width=1)), row=1, col=1)

    # Optional: mark last MACD cross on price
    if buy_signals:
        last_buy = buy_signals[-1][0]
        if last_buy in idx_plot:
            price_at_buy = df.loc[last_buy, '4. close']
            fig_price.add_trace(go.Scatter(
                x=[last_buy], y=[price_at_buy],
                mode="markers",
                marker=dict(symbol="triangle-up", color="#00B26F", size=12),
                name="Buy Xover"
            ), row=1, col=1)
    if sell_signals:
        last_sell = sell_signals[-1][0]
        if last_sell in idx_plot:
            price_at_sell = df.loc[last_sell, '4. close']
            fig_price.add_trace(go.Scatter(
                x=[last_sell], y=[price_at_sell],
                mode="markers",
                marker=dict(symbol="triangle-down", color="#E45756", size=12),
                name="Sell Xover"
            ), row=1, col=1)

    # Volume colored by up/down (bottom subplot)
    up_mask = df_plot['4. close'] >= df_plot['1. open']
    volume_colors = np.where(up_mask, "#C7F2D8", "#F7C9C6")
    fig_price.add_trace(
        go.Bar(x=idx_plot, y=df_plot['5. volume'], name="Volume", marker_color=volume_colors, opacity=0.8),
        row=2, col=1
    )

    # Legend below, margin increased to avoid overlap with title
    fig_price.update_layout(
        title=f"{symbol} — Price with SMA(20/50) & Bollinger Bands",
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Price"),
        xaxis2=dict(showgrid=False),
        yaxis2=dict(title="Volume"),
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=110),
        height=680
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # MACD (bottom)
    macd_idx = macd.index[-LOOKBACK_DAYS:]
    macd_plot = go.Figure()
    macd_plot.add_trace(go.Bar(
        x=macd_idx, y=hist.reindex(macd_idx),
        name="Histogram", marker_color="#9ecae1"
    ))
    macd_plot.add_trace(go.Scatter(
        x=macd_idx, y=macd.reindex(macd_idx),
        mode='lines', name='MACD', line=dict(color="#2ca02c", width=1.5)
    ))
    macd_plot.add_trace(go.Scatter(
        x=macd_idx, y=signal.reindex(macd_idx),
        mode='lines', name='Signal', line=dict(color="#d62728", width=1.5)
    ))
    # Buy/Sell markers within window
    bxs = [(t, v) for (t, v) in buy_signals if t in macd_idx]
    sxs = [(t, v) for (t, v) in sell_signals if t in macd_idx]
    if bxs:
        macd_plot.add_trace(go.Scatter(
            x=[t for (t, v) in bxs], y=[v for (t, v) in bxs],
            mode='markers', marker=dict(color='green', size=9), name='Buy Xover'
        ))
    if sxs:
        macd_plot.add_trace(go.Scatter(
            x=[t for (t, v) in sxs], y=[v for (t, v) in sxs],
            mode='markers', marker=dict(color='red', size=9), name='Sell Xover'
        ))
    macd_plot.update_layout(
        title="MACD",
        xaxis_title="Date", yaxis_title="Value",
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=50, b=110),
        height=420
    )
    st.plotly_chart(macd_plot, use_container_width=True)

    st.caption("⚠️ Educational use only. Not financial advice.")