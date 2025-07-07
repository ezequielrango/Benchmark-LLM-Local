import streamlit as st
import requests
import time
import psutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

st.set_page_config(layout="wide")

OLLAMA_URL = "http://localhost:11434" # URL del servidor Ollama. generalmente es esa. sino cambiar aquí. (no cree .env xq es local)

def find_ollama_pid():
    for p in psutil.process_iter(['pid', 'name']):
        try:
            if 'ollama' in p.info['name'].lower():
                return p.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None

def get_installed_models():
    try:
        res = requests.get(f"{OLLAMA_URL}/api/tags")
        if res.status_code == 200:
            data = res.json()
            return [m['name'] for m in data.get('models', [])]
    except:
        pass
    return []

# --- Benchmark ---
def run_benchmark(prompt, model, iterations):
    results = []
    ollama_pid = find_ollama_pid()
    if not ollama_pid:
        st.error("❌ No se encontró proceso Ollama corriendo en host.")
        return None

    proc = psutil.Process(ollama_pid)

    for i in range(iterations):
        with st.spinner(f"⏳ {model} → Iteración {i+1}/{iterations}..."):
            prompt_length = len(prompt.split())
            start = time.time()
            response = requests.post(f"{OLLAMA_URL}/api/generate", json={
                "model": model,
                "prompt": prompt,
                "stream": False
            })
            end = time.time()

            if response.status_code != 200:
                st.error(f"❌ Error: modelo '{model}' no existe o no responde.")
                return None

            elapsed = end - start
            data = response.json()
            text = data.get('response', '')
            tokens = len(text.split())
            tokens_per_sec = tokens / elapsed if elapsed > 0 else 0
            latency_per_token = elapsed / tokens if tokens > 0 else 0

            cpu_after = psutil.cpu_percent(interval=1)
            ram_after = proc.memory_info().rss / (1024 ** 2)
            swap_after = psutil.swap_memory().percent
            threads_after = proc.num_threads()

            results.append({
                'Model': model,
                'Iteration': i+1,
                'Prompt Length (tokens)': prompt_length,
                'Output Length (tokens)': tokens,
                'Total Time (s)': elapsed,
                'Tokens/s': tokens_per_sec,
                'Latency/token (s)': latency_per_token,
                'CPU (%)': cpu_after,
                'RAM (MB)': ram_after,
                'Swap (%)': swap_after,
                'Threads': threads_after
            })
    return results

# --- Guardar CSV ---
def save_csv(df, model):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_{model.replace(':','_')}_{timestamp}.csv"
    os.makedirs("benchmarks", exist_ok=True)
    path = os.path.join("benchmarks", filename)
    df.to_csv(path, index=False)
    return path

# --- Streamlit UI ---
st.title("🚀 Benchmark LLM Local Cross-OS")

available_models = get_installed_models()
if not available_models:
    st.warning("⚠️ No se encontraron modelos en Ollama. ¿Está corriendo?")
    st.stop()

with st.form("benchmark_form"):
    prompt = st.text_area("✏️ Prompt:", "Hello world por consola python simple y sencillo.")
    models = st.multiselect("🤖 Modelos a comparar:", available_models, default=available_models[:1])
    iterations = st.slider("🔁 Iteraciones:", min_value=1, max_value=10, value=3)
    submitted = st.form_submit_button("🚀 Ejecutar Benchmark")

if submitted:
    all_results = []
    st.markdown("""
    ## 📚 Índice de Resultados

    1️⃣ Estadísticas y gráficos individuales por modelo  
    2️⃣ Comparativa global entre modelos  
    3️⃣ Conclusión automática de costo-beneficio
    """)
    for model in models:
        results = run_benchmark(prompt, model, iterations)
        if results:
            df = pd.DataFrame(results)
            all_results.append(df)
            saved_path = save_csv(df, model)
            st.success(f"✅ Benchmark para **{model}** completado. CSV guardado en `{saved_path}`")

            st.subheader(f"📊 Estadísticas descriptivas para {model}:")
            st.write(df.describe().T)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader(f"📈 Boxplot Tokens/s ({model})")
                fig1, ax1 = plt.subplots(figsize=(4, 2))
                sns.boxplot(y=df['Tokens/s'], ax=ax1)
                st.pyplot(fig1, use_container_width=True)

                corr = df[['Prompt Length (tokens)', 'Output Length (tokens)', 'Total Time (s)',
                           'Tokens/s', 'Latency/token (s)', 'CPU (%)', 'RAM (MB)']].corr()

                st.subheader(f"📌 Correlación ({model})")
                fig_corr, ax_corr = plt.subplots(figsize=(4, 4))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm',
                            ax=ax_corr, linewidths=0.5, square=True,
                            cbar_kws={"shrink": 0.75})
                st.pyplot(fig_corr, use_container_width=True)

            with col2:
                st.subheader(f"📊 Histograma Latencia/token ({model})")
                fig2, ax2 = plt.subplots(figsize=(4, 2))
                sns.histplot(df['Latency/token (s)'], kde=True, ax=ax2)
                st.pyplot(fig2, use_container_width=True)

                st.subheader(f"📈 Evolución por Iteración ({model})")
                fig3, axs = plt.subplots(3, 2, figsize=(7, 5))
                axs[0, 0].plot(df['Iteration'], df['Tokens/s'], marker='o')
                axs[0, 0].set_title('Tokens/s')
                axs[0, 1].plot(df['Iteration'], df['CPU (%)'], marker='o', color='red')
                axs[0, 1].set_title('CPU (%)')
                axs[1, 0].plot(df['Iteration'], df['RAM (MB)'], marker='o', color='green')
                axs[1, 0].set_title('RAM (MB)')
                axs[1, 1].plot(df['Iteration'], df['Latency/token (s)'], marker='o', color='blue')
                axs[1, 1].set_title('Latencia/token')
                axs[2, 0].plot(df['Iteration'], df['Output Length (tokens)'], marker='o', color='orange')
                axs[2, 0].set_title('Output Length')
                axs[2, 1].plot(df['Iteration'], df['Threads'], marker='o', color='purple')
                axs[2, 1].set_title('Threads')
                for ax in axs.flat:
                    ax.set_xlabel('Iteración')
                    ax.grid(True)
                plt.tight_layout(h_pad=1)
                st.pyplot(fig3, use_container_width=True)

    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        st.subheader("📊 Comparativa global Tokens/s")
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.boxplot(x="Model", y="Tokens/s", data=combined_df, ax=ax)
        st.pyplot(fig)

        st.subheader("📊 Comparativa global Latencia/token")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        sns.boxplot(x="Model", y="Latency/token (s)", data=combined_df, ax=ax2)
        st.pyplot(fig2)

        st.subheader("📈 Evolución Tokens/s combinada")
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        sns.lineplot(data=combined_df, x="Iteration", y="Tokens/s", hue="Model", marker="o", ax=ax3)
        st.pyplot(fig3)

        st.subheader("📝 Conclusión sugerida")

        summary = combined_df.groupby('Model').agg({
            'Tokens/s': 'mean',
            'Latency/token (s)': 'mean'
        }).reset_index()
        summary['score'] = summary['Tokens/s'] / summary['Latency/token (s)']
        summary = summary.sort_values('score', ascending=False)
        best_model = summary.iloc[0]['Model']
        best_model_hardcore_test = "deepseek-r1:7b"  # Modelo hardcore para conclusión
        st.info(f"🔍 Modelo elegido para la conclusión automática: **{best_model}**")

        try:
            resumen_tabla = summary.to_markdown(index=False)
        except:
            resumen_tabla = summary.to_string(index=False)

        conclusion_prompt = f"""
        Analiza estos resultados:
        {resumen_tabla}

        ¿Cuál parece mejor? Explica en máximo 3 líneas. En español y claro.
        """
    
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": best_model_hardcore_test, "prompt": conclusion_prompt, "stream": False}
            )
            if response.status_code == 200:
                llm_answer = response.json().get('response', '')
                st.success(f"🤖 Conclusión generada por **{best_model_hardcore_test}**:\n\n{llm_answer}")
            else:
                raise Exception("LLM error")
        except Exception:
            st.warning("⚠️ No se pudo usar LLM. Fallback local:")
            st.write(summary[['Model', 'score']].to_string(index=False))
            st.success(f"✅ Según cálculo local: **{best_model}** es el mejor balanceado.")
