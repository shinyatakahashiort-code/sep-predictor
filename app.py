
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ページ設定
st.set_page_config(
    page_title="小児SE_p予測システム",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .prediction-result {
        font-size: 48px;
        font-weight: bold;
        color: #ff4b4b;
        text-align: center;
        padding: 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# タイトル
st.markdown('<p class="main-header">👁️ 小児のSE予測システム</p>', unsafe_allow_html=True)
st.markdown("---")

# モデル読み込み
@st.cache_resource
def load_model():
    try:
        model = joblib.load('sep_model.pkl')
        with open('model_info.json', 'r', encoding='utf-8') as f:
            info = json.load(f)
        return model, info
    except Exception as e:
        st.error(f"モデル読み込みエラー: {e}")
        return None, None

model, model_info = load_model()

if model is None:
    st.error("⚠️ モデルファイルが見つかりません。")
    st.stop()

# サイドバー - 患者情報入力
st.sidebar.header("📋 患者情報入力")

with st.sidebar:
    st.markdown("### 基本情報")
    patient_id = st.text_input("患者ID", value="", placeholder="例: P12345")
    
    age = st.number_input(
        "年齢 (歳)",
        min_value=1,
        max_value=100,
        value=8,
        step=1,
        help="患者の年齢を入力してください"
    )
    
    gender = st.selectbox(
        "性別",
        options=["女性", "男性"],
        help="患者の性別を選択してください"
    )
    
    eye_side = st.selectbox(
        "測定眼",
        options=["右眼", "左眼"],
        help="測定する眼を選択してください"
    )
    
    st.markdown("---")
    st.markdown("### 眼科測定値")
    
    k_avg = st.number_input(
        "K(AVG) - 角膜曲率平均",
        min_value=6.0,
        max_value=9.0,
        value=7.72,
        step=0.01,
        format="%.2f",
        help="角膜曲率の平均値"
    )
    
    al = st.number_input(
        "AL - 眼軸長 (mm)",
        min_value=20.0,
        max_value=32.0,
        value=24.22,
        step=0.01,
        format="%.2f",
        help="眼軸長の測定値"
    )
    
    lt = st.number_input(
        "LT - 水晶体厚 (mm)",
        min_value=2.0,
        max_value=6.0,
        value=3.44,
        step=0.01,
        format="%.2f",
        help="水晶体の厚さ"
    )
    
    acd = st.number_input(
        "ACD - 前房深度 (mm)",
        min_value=2.0,
        max_value=5.0,
        value=3.81,
        step=0.01,
        format="%.2f",
        help="前房の深さ"
    )
    
    st.markdown("---")
    predict_button = st.button("🔍 予測実行", use_container_width=True, type="primary")

# メインエリア
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📊 予測結果")
    
    if predict_button:
        # 性別を数値に変換
        gender_num = 0 if gender == "女性" else 1
        
        # 入力データ
        input_data = np.array([[age, gender_num, k_avg, al, lt, acd]])
        
        # 予測
        try:
            prediction = model.predict(input_data)[0]
            
            # 予測結果の表示
            st.markdown(f'<div class="prediction-result">予測 SE_p: {prediction:.2f} D</div>', 
                       unsafe_allow_html=True)
            
            # 予測の解釈
            if prediction < -3:
                interpretation = "⚠️ 強い近視傾向"
                color = "red"
            elif prediction < -1:
                interpretation = "📊 軽度近視傾向"
                color = "orange"
            elif prediction < 1:
                interpretation = "✅ 良好な範囲"
                color = "green"
            else:
                interpretation = "⚠️ 遠視傾向"
                color = "blue"
            
            st.markdown(f"### {interpretation}")
            
            # ゲージチャート
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = prediction,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "SE_p予測値", 'font': {'size': 24}},
                delta = {'reference': 0, 'increasing': {'color': "red"}},
                gauge = {
                    'axis': {'range': [-12, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': color},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [-12, -3], 'color': 'lightcoral'},
                        {'range': [-3, -1], 'color': 'lightyellow'},
                        {'range': [-1, 1], 'color': 'lightgreen'},
                        {'range': [1, 10], 'color': 'lightblue'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': prediction
                    }
                }
            ))
            
            fig.update_layout(
                height=400,
                font={'size': 16}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 入力データサマリー
            st.markdown("### 📝 入力データサマリー")
            summary_data = {
                '項目': ['患者ID', '年齢', '性別', '測定眼', 'K(AVG)', 'AL', 'LT', 'ACD'],
                '値': [
                    patient_id if patient_id else "-",
                    f"{age}歳",
                    gender,
                    eye_side,
                    f"{k_avg:.2f}",
                    f"{al:.2f} mm",
                    f"{lt:.2f} mm",
                    f"{acd:.2f} mm"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # 結果の保存
            if st.button("💾 結果を保存"):
                result_data = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'patient_id': patient_id,
                    'age': age,
                    'gender': gender,
                    'eye_side': eye_side,
                    'k_avg': k_avg,
                    'al': al,
                    'lt': lt,
                    'acd': acd,
                    'predicted_sep': float(prediction)
                }
                
                # CSVに保存
                result_df = pd.DataFrame([result_data])
                result_df.to_csv('predictions_history.csv', mode='a', 
                               header=not pd.io.common.file_exists('predictions_history.csv'),
                               index=False)
                st.success("✅ 予測結果を保存しました!")
            
        except Exception as e:
            st.error(f"予測エラー: {e}")
    else:
        st.info("👈 左のサイドバーから患者情報を入力し、「予測実行」ボタンをクリックしてください。")

with col2:
    st.markdown("### ℹ️ モデル情報")
    
    # モデル性能
    with st.expander("📈 モデル性能", expanded=True):
        perf = model_info['performance']
        st.metric("R² スコア", f"{perf['test_r2']:.4f}")
        st.metric("RMSE", f"{perf['test_rmse']:.4f}")
        st.metric("MAE", f"{perf['test_mae']:.4f}")
        st.caption(f"訓練サンプル数: {model_info['training_samples']}")
    
    # 特徴量重要度
    with st.expander("🔍 特徴量重要度", expanded=True):
        importance_df = pd.DataFrame({
            '特徴量': list(model_info['feature_importance'].keys()),
            '重要度': list(model_info['feature_importance'].values())
        }).sort_values('重要度', ascending=False)
        
        fig_importance = px.bar(
            importance_df,
            x='重要度',
            y='特徴量',
            orientation='h',
            title='特徴量の重要度',
            color='重要度',
            color_continuous_scale='Blues'
        )
        fig_importance.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # 使用方法
    with st.expander("📖 使用方法"):
        st.markdown("""
        **基本的な使い方:**
        1. サイドバーから患者情報を入力
        2. 眼科測定値を入力
        3. 「予測実行」ボタンをクリック
        4. 予測結果を確認
        
        **注意事項:**
        - すべての測定値を正確に入力してください
        - 予測結果は参考値です
        - 臨床判断は医師が行ってください
        """)

# バッチ予測機能
st.markdown("---")
st.markdown("### 📊 バッチ予測 (複数患者)")

uploaded_file = st.file_uploader(
    "Excelファイルをアップロード",
    type=['xlsx', 'xls'],
    help="患者データを含むExcelファイルをアップロードしてください"
)

if uploaded_file is not None:
    try:
        df_upload = pd.read_excel(uploaded_file)
        st.success(f"✅ {len(df_upload)}件のデータを読み込みました")
        
        # データプレビュー
        st.markdown("#### データプレビュー")
        st.dataframe(df_upload.head(10), use_container_width=True)
        
        if st.button("🚀 バッチ予測実行"):
            # 必要な列の確認
            required_cols = ['年齢', '性別', 'K(AVG)', 'AL', 'LT', 'ACD']
            
            # データ前処理
            df_pred = df_upload.copy()
            
            # 性別変換
            if df_pred['性別'].dtype == 'object':
                df_pred['性別'] = df_pred['性別'].map({'女性': 0, '男性': 1})
            
            # 予測
            predictions = []
            for idx, row in df_pred.iterrows():
                input_data = np.array([[
                    row['年齢'],
                    row['性別'],
                    row['K(AVG)'],
                    row['AL'],
                    row['LT'],
                    row['ACD']
                ]])
                pred = model.predict(input_data)[0]
                predictions.append(pred)
            
            df_pred['予測SE_p'] = predictions
            
            # 結果表示
            st.markdown("#### 予測結果")
            st.dataframe(df_pred, use_container_width=True)
            
            # ダウンロードボタン
            csv = df_pred.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 結果をダウンロード (CSV)",
                data=csv,
                file_name=f"sep_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv'
            )
            
    except Exception as e:
        st.error(f"ファイル処理エラー: {e}")

# フッター
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>SE_p予測システム v1.0</p>
    <p>⚠️ 本システムの予測結果は参考値です。最終的な臨床判断は医師が行ってください。</p>
</div>
""", unsafe_allow_html=True)
