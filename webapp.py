import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import random
from typing import List, Optional
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh

###########################
# Utility: Rolling Bollinger
###########################
def calculate_bollinger(df: pd.DataFrame, column: str, window: int = 20, stdev: int = 2) -> pd.DataFrame:
    """
    Compute Bollinger on `df[column]`, adding:
      - BB_mavg
      - BB_high
      - BB_low
    """
    df = df.copy()
    if column not in df.columns:
        return df  # or raise an error

    rolling_mean = df[column].rolling(window).mean()
    rolling_std  = df[column].rolling(window).std()

    df['BB_mavg'] = rolling_mean
    df['BB_high'] = rolling_mean + stdev * rolling_std
    df['BB_low']  = rolling_mean - stdev * rolling_std

    return df

###########################
# Helper: Make Normalized %
###########################
def compute_normalized_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a column 'pct_close' that is 0 at day0, and each subsequent day is
    100 * (Close[i] - Close[0]) / Close[0].
    """
    df = df.copy()
    if len(df) == 0:
        df['pct_close'] = np.nan
        return df
    base = df['Close'].iloc[0]
    df['pct_close'] = (df['Close'] - base) / base * 100.0
    return df

###########################
# Helper: Day-to-day increments
###########################
def compute_daily_incr(df: pd.DataFrame) -> pd.DataFrame:
    """
    If df has 'pct_close', add 'daily_incr' = pct_close[i] - pct_close[i-1].
    """
    df = df.copy()
    df['daily_incr'] = df['pct_close'].diff().fillna(0.0)
    return df

###########################
# Helper: RSI Calculation
###########################
def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) for the 'Close' prices.
    """
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

################################
# Default S&P 500 stock list
################################
DEFAULT_SP500_TOP20 = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "TSLA", "BRK-B", "JPM", "V", "JNJ",
    "WMT", "PG", "MA", "UNH", "HD",
    "BAC", "DIS", "XOM", "PFE", "VZ"
]

############################
# Data retrieval class
############################
class Data:
    """
    Pulls yfinance data for a list of equities set in __init__.
    If not set, defaults to 20 well-known S&P 500 stocks.
    """
    def __init__(self, symbols: Optional[List[str]] = None) -> None:
        self.symbols = symbols if symbols else DEFAULT_SP500_TOP20

    def get_90day_data(self, symbol: str) -> pd.DataFrame:
        """
        Fetch 90 days of daily data for a symbol.
        """
        df = yf.download(symbol, period="90d", interval="1d", progress=False)
        df.dropna(inplace=True)
        return df

    def get_current_price_data(self) -> pd.DataFrame:
        """
        Fetch the latest info for all symbols in self.symbols.
        Used for the marquee ticker.
        """
        data = []
        for symbol in self.symbols:
            ticker = yf.Ticker(symbol)
            info = ticker.history(period="2d")  # 2 days
            if not info.empty:
                last_close = info['Close'].iloc[-1]
                if len(info) > 1:
                    prev_close = info['Close'].iloc[-2]
                else:
                    prev_close = last_close
                change = 0.0
                if prev_close != 0:
                    change = (last_close - prev_close) / prev_close * 100
                data.append({
                    'Symbol': symbol,
                    'Price': last_close,
                    'Change%': change
                })
        return pd.DataFrame(data)

###############################
# WebApp Class
###############################
class WebApp:
    """
    Streamlit-based WebApp:
      - A top header bar (with partial transparency) contains the marquee ticker.
      - Main Page: 15 tiles (3 rows x 5 columns) for random symbols.
        Each tile:
          * Pulls 90-day data.
          * Computes 'pct_close' (normalized so that day0 is 0 on the selected lookback range).
          * Computes daily_incr.
          * Runs Bollinger on 'pct_close'.
          * Displays a combined chart with two subplots:
              - Top: daily_incr bars, a bold white price line, and Bollinger bands.
              - Bottom: RSI (14‑period, computed on raw 'Close' values) as an orange line.
            The two subplots share the same x‑axis with minimal spacing.
          * Uses a small header bubble.
          * Tile background color is based on the net price change.
      - Detail page: a larger version of the same chart.
      - Sidebar hidden by default.
    """
    def __init__(self, data_obj: Data):
        self.data_obj = data_obj
        if 'selected_symbol' not in st.session_state:
            st.session_state['selected_symbol'] = None

    def show_header_bar(self):
        """
        Create a fixed, semi-transparent header bar at the top with the marquee ticker.
        """
        ticker_df = self.data_obj.get_current_price_data()
        if ticker_df.empty:
            ticker_html = "<span>No Ticker Data Available.</span>"
        else:
            ticker_strs = []
            for _, row in ticker_df.iterrows():
                color = "darkgreen" if row['Change%'] >= 0 else "darkred"
                ticker_strs.append(
                    f"<span style='color:{color}; font-weight:bold;'>{row['Symbol']}: "
                    f"{row['Price']:.2f} ({row['Change%']:.2f}%)</span>"
                )
            ticker_html = " | ".join(ticker_strs)
        header_html = f"""
        <style>
        .header-bar {{
            background-color: rgba(0, 0, 0, 0.5);
            padding: 10px;
            text-align: center;
            font-size: 1.2em;
            color: white;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 9999;
        }}
        .body-container {{
            padding-top: 70px;
        }}
        </style>
        <div class="header-bar">
            <marquee>{ticker_html}</marquee>
        </div>
        """
        components.html(header_html, height=70)

    def build_pct_chart(self, df_90: pd.DataFrame, lookback_days: int) -> go.Figure:
        """
        Build a combined chart with:
          - Top subplot: the main chart with daily_incr bars, a bold white price line, and Bollinger bands.
          - Bottom subplot: the RSI (14‑period, computed on raw 'Close' values) as an orange line.
        Both subplots share the same x‑axis with minimal spacing.
        """
        # Process full data.
        df = df_90.copy()
        df = compute_normalized_price(df)
        df = compute_daily_incr(df)
        df = calculate_bollinger(df, column='pct_close', window=20, stdev=2)

        # Slice the last lookback_days.
        if len(df) > lookback_days:
            df = df.iloc[-lookback_days:].copy()

        # Create combined subplots with explicit row heights.
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.65, 0.35]
        )

        # Top subplot: main chart.
        bar_colors = ['green' if x >= 0 else 'red' for x in df['daily_incr']]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['daily_incr'],
                marker_color=bar_colors,
                opacity=0.7,
                name='Daily Incr'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['pct_close'],
                mode='lines',
                line=dict(color='white', width=4),
                name='Price'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_high'],
                mode='lines',
                line=dict(color='blue', width=3),
                name='BB High'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_low'],
                mode='lines',
                line=dict(color='blue', width=3),
                name='BB Low'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_mavg'],
                mode='lines',
                line=dict(color='lightblue', width=3),
                name='BB Mid'
            ),
            row=1, col=1
        )

        # Bottom subplot: RSI chart.
        # Use the provided snippet.
        rsi_df = df_90.copy()
        rsi_series = pd.DataFrame(compute_rsi(rsi_df, period=14)).dropna()
        rsi_series = rsi_series.iloc[-lookback_days:]
        fig.add_trace(
            go.Scatter(
                x=rsi_series.index,
                y=rsi_series[rsi_series.columns[0]],
                mode='lines',
                line=dict(color='orange', width=2),
                name='RSI'
            ),
            row=2, col=1
        )

        # Remove x-axis tick labels for the bottom subplot.
        fig.update_xaxes(showticklabels=False, row=2, col=1)

        # Remove y-axis titles.
        fig.update_yaxes(title_text="", row=1, col=1)
        fig.update_yaxes(title_text="", row=2, col=1, range=[0, 100])

        # Update layout with reduced overall height and zero left margin.
        fig.update_layout(
            showlegend=False,
            height=250,
            margin=dict(l=0, r=5, t=5, b=5),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        return fig

    def show_main_page(self):
        st.markdown("<div class='body-container'>", unsafe_allow_html=True)
        lookback_days = st.sidebar.slider("Days to Display", 7, 90, 25)

        all_syms = self.data_obj.symbols
        # For 5 columns wide and 3 rows, display up to 15 tiles.
        selected = random.sample(all_syms, min(15, len(all_syms)))
        num_columns = 5
        num_rows = (len(selected) + num_columns - 1) // num_columns

        for row in range(num_rows):
            cols = st.columns(num_columns)
            for col in range(num_columns):
                idx = row * num_columns + col
                if idx >= len(selected):
                    break
                symbol = selected[idx]
                with cols[col]:
                    df_90 = self.data_obj.get_90day_data(symbol)
                    if df_90.empty:
                        st.write(f"{symbol}: No Data")
                    else:
                        fig = self.build_pct_chart(df_90, lookback_days)

                        # Determine net change for header bubble.
                        df_sub = df_90 if len(df_90) < lookback_days else df_90.iloc[-lookback_days:]
                        # Using the .iloc[-1][0] fix.
                        first_close = df_sub['Close'].iloc[0][0]
                        last_close  = df_sub['Close'].iloc[-1][0]
                        net_pct = (last_close - first_close) / first_close * 100 if first_close != 0 else 0.0

                        if net_pct > 0:
                            tile_bg = "#3fa73f"  # green
                            ccolor  = "white"
                        elif net_pct < 0:
                            tile_bg = "#c04646"  # red
                            ccolor  = "white"
                        else:
                            tile_bg = "#888888"  # gray
                            ccolor  = "white"

                        st.markdown(
                            f"""
                            <div style="
                                background-color: {tile_bg};
                                border-radius: 5px;
                                box-shadow: 1px 1px 3px rgba(0,0,0,0.3);
                                padding: 5px;
                                margin-bottom: 2px;
                                text-align: center;
                            ">
                                <div style="font-size:0.9em; font-weight:bold; color:white;">
                                    {symbol}
                                </div>
                                <div style="font-size:0.8em; font-weight:bold; color:{ccolor};">
                                    {net_pct:.2f}%
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        if st.button(f"Details {symbol}", key=f"btn_{symbol}"):
                            st.session_state['selected_symbol'] = symbol
                            st.session_state['lookback_days'] = lookback_days
                            st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    def show_detail_page(self, symbol: str):
        st.markdown("<div class='body-container'>", unsafe_allow_html=True)
        st.title(f"Detail Page: {symbol}")
        lookback_days = st.session_state.get('lookback_days', 25)
        df_90 = self.data_obj.get_90day_data(symbol)
        if df_90.empty:
            st.write("No data.")
            return

        fig = self.build_pct_chart(df_90, lookback_days)
        st.plotly_chart(fig, use_container_width=True)

        if st.button("Back to Main"):
            st.session_state['selected_symbol'] = None
            st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    def run(self):
        st.set_page_config(
            page_title="Stock WebApp",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        # Auto-refresh every 30 seconds.
        st_autorefresh(interval=30000, key="autorefresh")
        self.show_header_bar()
        symbol = st.session_state.get('selected_symbol')
        if symbol:
            self.show_detail_page(symbol)
        else:
            self.show_main_page()

def main():
    data_obj = Data()
    app = WebApp(data_obj)
    app.run()

if __name__ == "__main__":
    main()