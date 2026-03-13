import sys
import os
import streamlit as st
import pandas as pd
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.visualization.plots import load_latest_benchmark_csv

def load_local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def generate_tier_models_html(models):
    """Generates the HTML markup for individual model pill cards inside a tier list."""
    html = ""
    for m in models:
        color = m.get('color', '#9CA3AF')
        html += f'''
        <div class="model-card">
            <div class="model-dot" style="background: {color};"></div>
            <span>{m['name']}</span>
            <span class="model-param">{m['params']}</span>
        </div>
        '''
    return html

def main():
    # Hide sidebar initially to match Onyx's full-width nav
    st.set_page_config(page_title="LLM Leaderboard", page_icon="🏆", layout="wide", initial_sidebar_state="collapsed")
    
    # Inject Precise CSS for Onyx Recreation
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        load_local_css(css_path)

    # Load local benchmark data to populate our actual models alongside S/A tier benchmarks
    df = load_latest_benchmark_csv(results_dir="results/benchmarks")
    local_models = []
    if df is not None and not df.empty:
        # Get unique models and assign them a generic B-tier or actual logic
        # Here we mock their extraction to fit the tier list seamlessly
        for m_name in df["Model"].unique():
            if "llama" in m_name.lower():
                color = "#10B981" # Greenish
                params = "8B"
            else:
                color = "#F59E0B" # Orange
                params = "7B"
            local_models.append({'name': m_name, 'color': color, 'params': params})
    
    # Render the Onyx Navbar exactly
    nav_html = """
    <div class="onyx-nav">
        <div class="onyx-logo">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L2 12L12 22L22 12L12 2Z" fill="#111827"/>
                <path d="M12 7L7 12L12 17L17 12L12 7Z" fill="#FFFFFF"/>
            </svg>
            onyx
        </div>
        <div class="onyx-menu">
            <span>Product ⌄</span>
            <span>Resources ⌄</span>
            <span>Company ⌄</span>
            <span>Pricing</span>
        </div>
        <div class="onyx-actions">
            <div style="display:flex; align-items:center; gap:6px; font-weight: 600; font-size: 0.9rem; color: #111827; background:#F3F4F6; padding:0.2rem 0.6rem; border-radius:12px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="#111827"><path d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.166 6.839 9.489.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.603-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.462-1.11-1.462-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.831.092-.646.35-1.086.636-1.336-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.564 9.564 0 0112 6.844c.85.004 1.705.114 2.504.336 1.909-1.294 2.747-1.025 2.747-1.025.546 1.379.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.138 20.161 22 16.416 22 12c0-5.523-4.477-10-10-10z"/></svg>
                17k
            </div>
            <div class="onyx-btn-outline">Try for Free</div>
            <div class="onyx-btn-solid">Book a Demo</div>
        </div>
    </div>
    """
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    # Tier arrays holding models to render
    s_tier_models = [
        {'name': 'Claude Opus 4.6', 'color': '#D97757', 'params': 'N/A'}, 
        {'name': 'GPT-5.4', 'color': '#10B981', 'params': 'N/A'}, 
        {'name': 'GLM-5', 'color': '#2563EB', 'params': '744B'}, 
        {'name': 'Kimi K2.5', 'color': '#1E3A8A', 'params': '1T'},
        {'name': 'DeepSeek V3.2', 'color': '#3B82F6', 'params': '685B'}
    ]
    
    a_tier_models = [
        {'name': 'Claude Sonnet 4.6', 'color': '#D97757', 'params': 'N/A'}, 
        {'name': 'Gemini 3.1 Pro', 'color': '#10B981', 'params': 'N/A'}, 
        {'name': 'Qwen 3.5', 'color': '#8B5CF6', 'params': '397B'},
        {'name': 'DeepSeek R1', 'color': '#3B82F6', 'params': '671B'},
        {'name': 'Mistral Large', 'color': '#F59E0B', 'params': '675B'},
        {'name': 'MiniMax M2.5', 'color': '#EF4444', 'params': '230B'},
        {'name': 'Step-3.5-Flash', 'color': '#8B5CF6', 'params': '196B'},
        {'name': 'MiMo-V2-Flash', 'color': '#F97316', 'params': '309B'}
    ]
    
    b_tier_models = [
        {'name': 'GPT-oss 120B', 'color': '#10B981', 'params': '117B'},
        {'name': 'Nemotron Ultra 253B', 'color': '#65A30D', 'params': '253B'}
    ]
    
    # Merge local run models into the B-tier
    b_tier_models.extend(local_models)
    
    # Raw HTML for Main Body mapping the tiers
    main_html = f"""
    <div class="main-content">
        <div class="hero-subtitle">Best LLMs — 2026 Rankings</div>
        <div class="hero-title">LLM Leaderboard</div>
        <div class="hero-desc">The definitive ranking of every major LLM — open and closed source — compared across reasoning, coding, math, agentic, software engineering, and chat benchmarks.</div>
        
        <div class="author-line">
            <img src="https://ui-avatars.com/api/?name=AI&background=111827&color=fff&rounded=true" width="22" height="22" style="border-radius: 50%;"/>
            <span>AI Researcher · Last updated: {date_str}</span>
        </div>
        
        <div class="pill-container">
            <div class="pill-btn">Open Source LLM Leaderboard <span style="color:#9CA3AF;">→</span></div>
            <div class="pill-btn">Self-Hosted LLM Leaderboard <span style="color:#9CA3AF;">→</span></div>
            <div class="pill-btn">Best LLM for Coding <span style="color:#9CA3AF;">→</span></div>
        </div>
        
        <div class="tabs-container">
            <div class="tab-group">
                <div class="tab active">Overall</div>
                <div class="tab">Coding</div>
                <div class="tab">Math</div>
                <div class="tab">Chat</div>
                <div class="tab">Reasoning</div>
                <div class="tab">Agentic</div>
            </div>
            <div class="filter-group">
                <div class="filter-btn">Medium</div>
                <div class="filter-btn">Large</div>
            </div>
        </div>
        
        <div class="tier-list">
            <div class="tier-row">
                <div class="tier-label tier-s">S</div>
                <div class="tier-models">
                    {generate_tier_models_html(s_tier_models)}
                </div>
            </div>
            <div class="tier-row">
                <div class="tier-label tier-a">A</div>
                <div class="tier-models">
                    {generate_tier_models_html(a_tier_models)}
                </div>
            </div>
            <div class="tier-row">
                <div class="tier-label tier-b">B</div>
                <div class="tier-models">
                    {generate_tier_models_html(b_tier_models)}
                </div>
            </div>
        </div>
    </div>
    """
    
    # Inject Layout
    st.markdown(nav_html, unsafe_allow_html=True)
    st.markdown(main_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
