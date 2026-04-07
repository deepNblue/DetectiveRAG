"""
Boot Animation — 忒修斯之线 (Ariadne's Thread)
v5: Gradio-native dismiss via JS event, auto-hide via load event, proper reveal
"""
import base64
import os

_SPLASH_IMG_PATH = os.path.join(os.path.dirname(__file__), "splash_image.webp")


def _get_image_base64() -> str:
    """加载启动图并转为base64"""
    path = _SPLASH_IMG_PATH
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(__file__), "splash_image.jpg")
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        data = f.read()
    ext = "png" if path.endswith(".png") else "jpeg"
    return f"data:image/{ext};base64,{base64.b64encode(data).decode()}"


def get_boot_animation_html() -> str:
    """返回片头动画HTML（纯CSS/HTML，无inline JS — JS由Gradio事件驱动）"""
    img_b64 = _get_image_base64()
    if not img_b64:
        img_b64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="

    return f'''<!-- ══════════════════════════════════════════════════════════ -->
<!--  忒修斯之线 · 启动片头  v5  — JS由Gradio事件驱动          -->
<!-- ══════════════════════════════════════════════════════════ -->
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&family=Share+Tech+Mono&display=swap');

/* ── Splash root ── */
#splash-overlay {{
  position: fixed;
  inset: 0;
  z-index: 99998;
  background: #050508;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  animation: splash_in 0.6s ease-out forwards;
}}

/* 背景照片 */
.splash-bg {{
  position: absolute;
  inset: 0;
  background-image: url('{img_b64}');
  background-size: cover;
  background-position: center;
  filter: brightness(0.68) saturate(0.8) contrast(1.0);
  animation: bg_zoom 9s ease-in-out forwards;
}}

/* CRT扫描线 */
.splash-scanlines {{
  position: absolute;
  inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,0,0,0.15) 2px,
    rgba(0,0,0,0.15) 4px
  );
  pointer-events: none;
  z-index: 2;
}}

/* 暗角 */
.splash-vignette {{
  position: absolute;
  inset: 0;
  background: radial-gradient(ellipse at 50% 50%,
    rgba(0,0,0,0.0) 0%,
    rgba(0,0,0,0.35) 60%,
    rgba(0,0,0,0.7) 100%
  );
  pointer-events: none;
  z-index: 3;
}}

/* 顶部光柱 */
.splash-lightbar {{
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, rgba(200,150,60,0.9), transparent);
  animation: lightbar_sweep 2.5s ease-in-out infinite;
  z-index: 4;
}}

/* 红色脉冲边框 */
.splash-border {{
  position: absolute;
  inset: 0;
  border: 1px solid rgba(180,60,40,0.0);
  animation: border_pulse 2s ease-in-out infinite;
  z-index: 4;
  pointer-events: none;
}}

/* 中央内容 */
.splash-content {{
  position: relative;
  z-index: 10;
  text-align: center;
  padding: 0 2rem;
  user-select: none;
}}

.splash-case-tag {{
  font-family: 'Share Tech Mono', monospace;
  font-size: 11px;
  letter-spacing: 7px;
  color: rgba(200,150,60,0.6);
  text-transform: uppercase;
  margin-bottom: 20px;
  animation: fade_up 1s 0.3s both ease-out;
}}

.splash-title {{
  font-family: 'Noto Serif SC', 'SimHei', serif;
  font-size: clamp(48px, 9vw, 96px);
  font-weight: 700;
  color: #ECD89A;
  letter-spacing: 14px;
  text-shadow:
    0 0 20px rgba(210,160,70,0.6),
    0 0 60px rgba(200,130,40,0.3),
    0 2px 4px rgba(0,0,0,0.9);
  margin: 0 0 6px;
  white-space: nowrap;
  border-right: 2px solid rgba(220,170,80,0.85);
  animation: blink_caret 0.55s step-end infinite;
}}

/* 每个字逐个出现，位置固定不移动 */
.splash-title .tw {{
  opacity: 0;
  display: inline-block;
  animation: char_reveal 0.35s forwards;
}}

@keyframes char_reveal {{
  0%   {{ opacity: 0; transform: translateY(8px); }}
  60%  {{ opacity: 1; transform: translateY(-2px); }}
  100% {{ opacity: 1; transform: translateY(0); }}
}}

.splash-subtitle {{
  font-family: 'Noto Serif SC', 'SimHei', serif;
  font-size: clamp(13px, 2vw, 19px);
  color: rgba(210,180,110,0.6);
  letter-spacing: 14px;
  margin-top: 4px;
  animation: fade_up 1s 2.8s both ease-out;
}}

.splash-divider {{
  width: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(200,150,60,0.5), transparent);
  margin: 22px auto;
  animation: expand_x 1s 2.1s both ease-out;
}}

.splash-quote {{
  font-family: 'Noto Serif SC', 'SimHei', serif;
  font-size: 13px;
  color: rgba(200,175,110,0.4);
  letter-spacing: 2px;
  font-style: italic;
  margin-bottom: 48px;
  animation: fade_up 1s 3.2s both ease-out;
}}

/* 进度条 */
.splash-progress-wrap {{
  position: absolute;
  bottom: 72px;
  left: 50%;
  transform: translateX(-50%);
  width: min(340px, 78vw);
  animation: fade_up 0.8s 3.6s both ease-out;
  z-index: 10;
}}

.splash-progress-bg {{
  width: 100%;
  height: 2px;
  background: rgba(200,150,60,0.12);
  border-radius: 2px;
  overflow: hidden;
}}

.splash-progress-bar {{
  height: 100%;
  background: linear-gradient(90deg,
    rgba(200,140,50,0.5) 0%,
    rgba(230,170,60,0.95) 60%,
    rgba(255,200,80,1) 100%
  );
  border-radius: 2px;
  box-shadow: 0 0 10px rgba(240,170,50,0.7);
  width: 0%;
  animation: progress_fill 6s linear 3.6s forwards;
}}

.splash-progress-label {{
  text-align: center;
  font-family: 'Share Tech Mono', monospace;
  font-size: 9px;
  color: rgba(200,150,60,0.35);
  letter-spacing: 5px;
  margin-top: 10px;
}}

/* 跳过按钮（由Gradio组件替代，这里仅占位） */
.splash-skip-btn-hook {{
  position: absolute;
  bottom: 56px;
  right: 36px;
  z-index: 20;
}}

/* 版本标签 */
.splash-version {{
  position: absolute;
  top: 22px;
  right: 26px;
  font-family: 'Share Tech Mono', monospace;
  font-size: 10px;
  color: rgba(200,150,60,0.22);
  letter-spacing: 2px;
  z-index: 20;
  animation: fade_up 0.8s 1.5s both ease-out;
}}

/* ── 关键帧 ── */
@keyframes splash_in {{
  from {{ opacity: 0; }}
  to   {{ opacity: 1; }}
}}

@keyframes bg_zoom {{
  from {{ transform: scale(1.1); filter: brightness(0.3) saturate(0.65); }}
  to   {{ transform: scale(1.0); filter: brightness(0.68) saturate(0.8) contrast(1.0); }}
}}

@keyframes blink_caret {{
  50% {{ border-color: transparent; }}
}}

@keyframes fade_up {{
  from {{ opacity: 0; transform: translateY(14px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}

@keyframes expand_x {{
  from {{ width: 0; opacity: 0; }}
  to   {{ width: 140px; opacity: 1; }}
}}

@keyframes progress_fill {{
  from {{ width: 0%; }}
  to   {{ width: 100%; }}
}}

@keyframes lightbar_sweep {{
  0%   {{ opacity: 0; transform: scaleX(0) translateY(0); }}
  25%  {{ opacity: 1; transform: scaleX(1) translateY(0); }}
  75%  {{ opacity: 0.5; transform: scaleX(1) translateY(0); }}
  100% {{ opacity: 0; transform: scaleX(1) translateY(0); }}
}}

@keyframes border_pulse {{
  0%, 100% {{ border-color: rgba(180,60,40,0.0); box-shadow: inset 0 0 0 rgba(180,60,40,0); }}
  50%       {{ border-color: rgba(180,60,40,0.2); box-shadow: inset 0 0 20px rgba(180,60,40,0.05); }}
}}

/* 消失动画 */
#splash-overlay.splash-out {{
  animation: splash_out 0.9s ease-in forwards;
}}

@keyframes splash_out {{
  0%   {{ opacity: 1; transform: scale(1.0); }}
  40%  {{ opacity: 1; transform: scale(1.01); }}
  100% {{ opacity: 0; transform: scale(1.03); visibility: hidden; pointer-events: none; }}
}}
</style>

<!-- Overlay -->
<div id="splash-overlay">
  <div class="splash-bg"></div>
  <div class="splash-scanlines"></div>
  <div class="splash-vignette"></div>
  <div class="splash-lightbar"></div>
  <div class="splash-border"></div>

  <div class="splash-content">
    <div class="splash-case-tag">CASE NO. TH-2026-001</div>
    <h1 class="splash-title"><span class="tw" style="animation-delay:0.9s">忒</span><span class="tw" style="animation-delay:1.15s">修</span><span class="tw" style="animation-delay:1.4s">斯</span><span class="tw" style="animation-delay:1.65s">之</span><span class="tw" style="animation-delay:1.9s">线</span></h1>
    <div class="splash-subtitle">多模态 RAG · 双线并行推理系统</div>
    <div class="splash-divider"></div>
    <div class="splash-quote">"真相只有一个——在证据的最深处"</div>
  </div>

  <!-- 进度条 -->
  <div class="splash-progress-wrap">
    <div class="splash-progress-bg">
      <div class="splash-progress-bar" id="splash-progress"></div>
    </div>
    <div class="splash-progress-label">LOADING · INITIALIZING · ANALYZING</div>
  </div>

  <!-- 跳过按钮由Gradio组件渲染到这里 -->
  <div class="splash-skip-btn-hook" id="splash-skip-hook"></div>

  <div class="splash-version">v14.0 · ARIADNE_THREAD</div>
</div>
'''
