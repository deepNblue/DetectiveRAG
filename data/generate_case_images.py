#!/usr/bin/env python3
"""
生成多模态案件的图片资源
- 案发现场平面图
- 证据照片（模拟）
- 监控截图（模拟）
"""
import os, math
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    os.system("pip install Pillow -q")
    from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(__file__), "case_images")
os.makedirs(OUT, exist_ok=True)

# 颜色方案 - 赛博侦探风
BG = (15, 15, 22)
GRID = (30, 30, 40)
WALL = (80, 80, 95)
TEXT = (200, 200, 210)
ACCENT = (0, 255, 65)
WARN = (255, 100, 100)
BLUE = (70, 130, 230)
YELLOW = (255, 210, 60)

def get_font(size=16):
    """获取支持中文的字体"""
    font_paths = [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]
    for p in font_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except:
                pass
    return ImageFont.load_default()


def draw_room(draw, x, y, w, h, label="", color=WALL):
    """画房间"""
    draw.rectangle([x, y, x+w, y+h], outline=color, width=2)
    if label:
        font = get_font(14)
        draw.text((x + w//2 - len(label)*7, y + 6), label, fill=TEXT, font=font)


def draw_person(draw, x, y, label="", color=ACCENT, dead=False):
    """画人物标记"""
    r = 8
    draw.ellipse([x-r, y-r, x+r, y+r], fill=color, outline=WARN if dead else ACCENT, width=2)
    if dead:
        draw.line([x-r, y-r, x+r, y+r], fill=WARN, width=2)
        draw.line([x-r, y+r, x+r, y-r], fill=WARN, width=2)
    if label:
        font = get_font(11)
        draw.text((x + 12, y - 8), label, fill=color, font=font)


def draw_evidence_marker(draw, x, y, num, label=""):
    """画证据标记 (编号圆)"""
    r = 10
    draw.ellipse([x-r, y-r, x+r, y+r], fill=YELLOW, outline=(0,0,0), width=1)
    font = get_font(11)
    draw.text((x - 4, y - 7), str(num), fill=(0,0,0), font=font)
    if label:
        draw.text((x + 14, y - 6), label, fill=YELLOW, font=font)


# ============================================================
# 图片1: CASE-001 密室案发现场平面图
# ============================================================
def gen_case001_scene():
    W, H = 800, 600
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    font_title = get_font(20)
    font_sm = get_font(12)

    # 标题
    d.text((W//2 - 160, 10), "📍 案发现场平面图 — 陈伯远别墅地下室", fill=ACCENT, font=font_title)

    # 网格背景
    for i in range(0, W, 40):
        d.line([(i, 0), (i, H)], fill=GRID)
    for j in range(0, H, 40):
        d.line([(0, j), (W, j)], fill=GRID)

    # 密室主体 (大房间)
    rx, ry, rw, rh = 100, 80, 500, 350
    d.rectangle([rx, ry, rx+rw, ry+rh], outline=WARN, width=3)
    d.text((rx + 10, ry + 10), "🔒 密室 (电子密码锁)", fill=WARN, font=get_font(16))

    # 密室门
    d.rectangle([rx+rw-3, ry+140, rx+rw+8, ry+200], fill=WARN)
    d.text((rx+rw+12, ry+155), "密码门", fill=WARN, font=font_sm)

    # 书桌 + 死者
    desk_x, desk_y = 250, 200
    d.rectangle([desk_x, desk_y, desk_x+120, desk_y+60], fill=(40, 40, 55), outline=BLUE, width=2)
    d.text((desk_x+20, desk_y+5), "红木书桌", fill=BLUE, font=font_sm)
    draw_person(d, desk_x+60, desk_y+30, "死者:陈伯远", color=WARN, dead=True)

    # 普洱茶 (证据1)
    draw_evidence_marker(d, desk_x+130, desk_y+10, 1, "普洱茶(含氰化钾)")

    # 古籍
    d.text((desk_x+10, desk_y+35), "📖 古籍", fill=(180,180,190), font=font_sm)

    # 空调系统 (证据3)
    d.rectangle([rx+20, ry+260, rx+160, ry+310], fill=(30,50,30), outline=ACCENT, width=2)
    d.text((rx+30, ry+270), "❄️ 空调系统", fill=ACCENT, font=font_sm)
    draw_evidence_marker(d, rx+170, ry+285, 3, "滤网被更换")

    # 通风管道
    d.line([(rx+160, ry+285), (rx+rw-30, ry+285)], fill=(60,60,80), width=3)
    d.text((rx+250, ry+268), "通风管道(10cm)", fill=(100,100,120), font=font_sm)

    # 空调管道内壁残留 (证据4)
    draw_evidence_marker(d, rx+250, ry+305, 4, "管道内壁氰化物残留")

    # 走廊 (密室到储物间)
    d.rectangle([rx+rw+8, ry+140, rx+rw+80, ry+260], outline=WALL, width=2)
    d.text((rx+rw+14, ry+180), "走\n廊", fill=TEXT, font=get_font(14))

    # 储物间
    draw_room(d, rx+rw+80, ry+100, 120, 200, "储物间", BLUE)
    draw_evidence_marker(d, rx+rw+140, ry+200, 6, "被拆药瓶")

    # 管家房
    draw_room(d, 100, 460, 200, 100, "管家王福来房", BLUE)
    draw_person(d, 200, 510, "管家王福来", color=ACCENT)
    draw_evidence_marker(d, 280, 500, 7, "古董变卖账本")

    # 一楼客厅
    draw_room(d, 350, 460, 180, 100, "一楼客厅", BLUE)
    draw_person(d, 440, 510, "赵天成", color=(130,180,130))

    # 监控摄像头标记
    d.text((rx+rw+40, ry+135), "📹", fill=ACCENT, font=get_font(16))

    # 图例
    d.text((600, 460), "图例:", fill=TEXT, font=get_font(14))
    draw_person(d, 615, 490, "活人", color=ACCENT)
    draw_person(d, 615, 520, "死者", color=WARN, dead=True)
    draw_evidence_marker(d, 615, 550, "N", "证据编号")

    path = os.path.join(OUT, "case001_scene_plan.png")
    img.save(path)
    print(f"✅ {path}")
    return path


# ============================================================
# 图片2: CASE-001 证据照片 — 茶杯+药瓶
# ============================================================
def gen_case001_evidence():
    W, H = 600, 400
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    font_title = get_font(18)
    font_sm = get_font(12)

    d.text((W//2-120, 10), "🔬 证据特写 — 案件 CASE-001", fill=ACCENT, font=font_title)

    # 左侧: 茶杯
    cx, cy = 150, 200
    # 杯身
    d.polygon([(cx-40, cy-60), (cx-30, cy+40), (cx+30, cy+40), (cx+40, cy-60)],
              fill=(60,40,20), outline=BLUE, width=2)
    # 液面
    d.ellipse([cx-38, cy-40, cx+38, cy-20], fill=(80,50,25))
    # 热气
    for i in range(3):
        x = cx - 15 + i*15
        d.arc([x, cy-90, x+12, cy-60], 0, 360, fill=(100,100,120), width=1)

    d.text((cx-50, cy+60), "普洱茶", fill=TEXT, font=get_font(14))
    d.text((cx-70, cy+80), "检出氰化钾", fill=WARN, font=font_sm)
    d.text((cx-70, cy+100), "50mg/L", fill=WARN, font=get_font(16))
    # 指纹标记
    d.text((cx+50, cy-30), "👆 指纹:", fill=YELLOW, font=font_sm)
    d.text((cx+50, cy-12), "陈伯远+王福来", fill=YELLOW, font=font_sm)

    # 右侧: 药瓶
    bx, by = 430, 190
    d.rectangle([bx-25, by-70, bx+25, by+50], fill=(40,60,40), outline=ACCENT, width=2)
    d.rectangle([bx-20, by-65, bx+20, by-40], fill=(60,90,60))
    d.text((bx-15, by-60), "Rx", fill=ACCENT, font=get_font(14))
    # 标签被撕
    d.rectangle([bx-25, by-10, bx+25, by+30], fill=(50,50,60), outline=YELLOW, width=2)
    d.text((bx-22, by-5), "标签已撕", fill=YELLOW, font=font_sm)
    d.text((bx-22, by+12), "开瓶状态", fill=YELLOW, font=font_sm)

    d.text((bx-50, by+70), "储物间发现", fill=TEXT, font=get_font(14))
    d.text((bx-50, by+90), "被拆开的药瓶", fill=WARN, font=font_sm)

    # 分隔线
    d.line([(300, 60), (300, 370)], fill=WALL, width=1)

    path = os.path.join(OUT, "case001_evidence_photo.png")
    img.save(path)
    print(f"✅ {path}")
    return path


# ============================================================
# 图片3: 新案件 IMG-001 — 监控截图（模拟）
# ============================================================
def gen_monitor_screenshot():
    W, H = 640, 480
    img = Image.new("RGB", (W, H), (10, 10, 15))
    d = ImageDraw.Draw(img)

    # 监控画面效果 - 偏绿
    # 天花板视角的走廊
    d.rectangle([40, 40, W-40, H-60], fill=(15, 20, 18), outline=(0,80,0), width=1)

    # 走廊线条
    for i in range(5):
        y = 100 + i * 60
        d.line([(40, y), (W-40, y)], fill=(20, 40, 25), width=1)

    # 人形轮廓 (模糊)
    # 人物A
    d.ellipse([200, 140, 240, 180], fill=(30, 50, 35), outline=(0, 100, 0), width=2)
    d.rectangle([195, 180, 245, 280], fill=(30, 50, 35), outline=(0, 100, 0), width=2)
    d.text((190, 290), "21:32:15", fill=(0, 200, 0), font=get_font(12))
    d.text((250, 220), "? 未知人物A", fill=YELLOW, font=get_font(13))

    # 人物B (较远)
    d.ellipse([420, 160, 448, 185], fill=(30, 50, 35), outline=(0, 100, 0), width=2)
    d.rectangle([416, 185, 452, 260], fill=(30, 50, 35), outline=(0, 100, 0), width=2)
    d.text((460, 200), "? 未知人物B", fill=YELLOW, font=get_font(13))

    # 时间戳
    d.text((W-250, H-45), "CAM-03  2024-03-15 21:32", fill=(0, 200, 0), font=get_font(14))
    d.text((50, H-45), "REC ●", fill=WARN, font=get_font(14))

    # 标题
    d.text((50, 12), "📹 监控录像截图 — 别墅走廊 CAM-03", fill=ACCENT, font=get_font(16))

    # 噪点效果
    import random
    random.seed(42)
    for _ in range(200):
        nx = random.randint(40, W-40)
        ny = random.randint(40, H-60)
        c = random.randint(15, 30)
        d.point((nx, ny), fill=(c, c+10, c))

    path = os.path.join(OUT, "case001_monitor_cam03.png")
    img.save(path)
    print(f"✅ {path}")
    return path


# ============================================================
# 图片4: 新案件 IMG-002 — 指纹对比图
# ============================================================
def gen_fingerprint_comparison():
    W, H = 700, 350
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    d.text((W//2-100, 10), "🔍 指纹比对报告", fill=ACCENT, font=get_font(18))

    # 左指纹 (现场提取)
    d.text((80, 50), "现场提取指纹", fill=TEXT, font=get_font(14))
    d.ellipse([70, 80, 220, 230], outline=BLUE, width=2)
    # 指纹纹路 (简化)
    import random
    random.seed(123)
    cx, cy = 145, 155
    for i in range(12):
        r = 10 + i * 5
        angle_s = random.uniform(0, 0.5)
        d.arc([cx-r, cy-r, cx+r, cy+r], int(angle_s*360), int((angle_s+0.7)*360),
              fill=(80, 120, 180), width=1)

    # 中间: 匹配标记
    d.text((255, 130), "VS", fill=WARN, font=get_font(24))
    d.text((230, 170), "匹配度", fill=TEXT, font=get_font(14))
    d.text((235, 195), "98.7%", fill=ACCENT, font=get_font(20))

    # 右指纹 (嫌疑人)
    d.text((400, 50), "嫌疑人指纹样本", fill=TEXT, font=get_font(14))
    d.ellipse([390, 80, 540, 230], outline=ACCENT, width=2)
    cx2, cy2 = 465, 155
    random.seed(456)
    for i in range(12):
        r = 10 + i * 5
        angle_s = random.uniform(0, 0.5)
        d.arc([cx2-r, cy2-r, cx2+r, cy2+r], int(angle_s*360), int((angle_s+0.7)*360),
              fill=(0, 180, 50), width=1)

    # 结论
    d.rectangle([50, 260, W-50, 320], fill=(30, 15, 15), outline=WARN, width=2)
    d.text((70, 275), "⚠️ 结论: 现场指纹与嫌疑人【王福来】右手拇指匹配 (98.7%)", fill=WARN, font=get_font(14))

    path = os.path.join(OUT, "case002_fingerprint.png")
    img.save(path)
    print(f"✅ {path}")
    return path


# ============================================================
# 图片5: 新案件 IMG-003 — 手机短信截图
# ============================================================
def gen_phone_screenshot():
    W, H = 380, 600
    img = Image.new("RGB", (W, H), (20, 20, 25))
    d = ImageDraw.Draw(img)

    # 手机外框
    d.rectangle([10, 10, W-10, H-10], outline=(60,60,70), width=2, fill=(25, 25, 30))

    # 状态栏
    d.text((30, 20), "21:45", fill=(180,180,180), font=get_font(14))
    d.text((W-80, 20), "🔋 87%", fill=(180,180,180), font=get_font(12))

    # 聊天标题
    d.rectangle([20, 45, W-20, 80], fill=(35, 35, 42))
    d.text((W//2-40, 52), "林若华医生", fill=TEXT, font=get_font(14))

    # 消息气泡
    # 发送的消息 (右侧, 绿色)
    msg_y = 110
    d.rounded_rectangle([130, msg_y, W-35, msg_y+45], radius=8, fill=(0, 80, 40))
    d.text((140, msg_y+5), "林医生，药已服用", fill=ACCENT, font=get_font(13))
    d.text((140, msg_y+25), "谢谢关心", fill=(100,200,100), font=get_font(12))
    d.text((W-80, msg_y+48), "21:45", fill=(100,100,110), font=get_font(10))

    # 对方回复 (左侧, 灰色)
    msg_y2 = 180
    d.rounded_rectangle([35, msg_y2, 250, msg_y2+30], radius=8, fill=(45, 45, 55))
    d.text((45, msg_y2+8), "好的，注意休息", fill=(160,160,170), font=get_font(13))
    d.text((40, msg_y2+33), "21:43", fill=(100,100,110), font=get_font(10))

    # 更早的消息
    msg_y3 = 240
    d.rounded_rectangle([35, msg_y3, 260, msg_y3+50], radius=8, fill=(45, 45, 55))
    d.text((45, msg_y3+5), "陈先生，今晚记得", fill=(160,160,170), font=get_font(13))
    d.text((45, msg_y3+25), "按时服药", fill=(160,160,170), font=get_font(13))
    d.text((40, msg_y3+53), "20:30", fill=(100,100,110), font=get_font(10))

    msg_y4 = 320
    d.rounded_rectangle([130, msg_y4, W-35, msg_y4+45], radius=8, fill=(0, 80, 40))
    d.text((140, msg_y4+5), "茶泡好了，准备去", fill=ACCENT, font=get_font(13))
    d.text((140, msg_y4+25), "密室看书", fill=ACCENT, font=get_font(13))
    d.text((W-80, msg_y4+48), "21:28", fill=(100,100,110), font=get_font(10))

    # 重点标记 - 最后一条短信
    d.rectangle([20, 90, W-20, 170], outline=WARN, width=2)
    d.text((25, 92), "⚠️ 最后发送的短信", fill=WARN, font=get_font(10))

    # 底部输入框
    d.rectangle([25, H-60, W-25, H-30], outline=(50,50,60), width=1, fill=(30,30,35))
    d.text((35, H-52), "输入消息...", fill=(80,80,90), font=get_font(12))

    path = os.path.join(OUT, "case001_phone_sms.png")
    img.save(path)
    print(f"✅ {path}")
    return path


# ============================================================
# 图片6: 新案件 IMG-004 — 毒物分析报告
# ============================================================
def gen_toxicology_report():
    W, H = 700, 500
    img = Image.new("RGB", (W, H), (252, 252, 248))
    d = ImageDraw.Draw(img)

    # 正式报告风格
    d.rectangle([0, 0, W, 50], fill=(30, 30, 40))
    d.text((W//2-100, 12), "📋 法医毒物分析报告", fill=ACCENT, font=get_font(18))

    # 报告头
    d.text((30, 65), "报告编号: FG-2024-0315-001", fill=(60,60,60), font=get_font(12))
    d.text((30, 85), "委托单位: 市公安局刑侦支队", fill=(60,60,60), font=get_font(12))
    d.text((30, 105), "死者: 陈伯远  男  62岁", fill=(60,60,60), font=get_font(12))
    d.text((400, 65), "检验日期: 2024-03-16", fill=(60,60,60), font=get_font(12))
    d.text((400, 85), "报告日期: 2024-03-17", fill=(60,60,60), font=get_font(12))

    # 分割线
    d.line([(20, 130), (W-20, 130)], fill=(150,150,150), width=1)

    # 检验结果表格
    d.text((30, 140), "检验结果", fill=(30,30,30), font=get_font(16))

    # 表头
    y = 170
    d.rectangle([30, y, W-30, y+25], fill=(40, 40, 55))
    d.text((40, y+4), "检验项目", fill=TEXT, font=get_font(12))
    d.text((220, y+4), "检验结果", fill=TEXT, font=get_font(12))
    d.text((420, y+4), "正常值", fill=TEXT, font=get_font(12))
    d.text((560, y+4), "异常", fill=TEXT, font=get_font(12))

    # 数据行
    rows = [
        ("血液氰化物浓度", "3.2 μg/mL", "< 0.5 μg/mL", "⚠️ 超标6.4倍"),
        ("茶水中氰化钾", "50 mg/L", "—", "⚠️ 检出"),
        ("胃内容物氰化物", "2.8 μg/mL", "< 0.3 μg/mL", "⚠️ 超标9.3倍"),
        ("空调滤网残留", "检出氰化物", "—", "⚠️ 检出"),
        ("空调管道内壁", "微量氰化物", "—", "⚠️ 检出"),
        ("酒精浓度", "0.02%", "< 0.02%", "正常"),
    ]
    for i, (item, result, normal, note) in enumerate(rows):
        ry = y + 28 + i * 28
        bg = (245, 245, 240) if i % 2 == 0 else (235, 235, 230)
        d.rectangle([30, ry, W-30, ry+26], fill=bg)
        d.text((40, ry+5), item, fill=(40,40,40), font=get_font(12))
        d.text((220, ry+5), result, fill=(40,40,40), font=get_font(12))
        d.text((420, ry+5), normal, fill=(100,100,100), font=get_font(12))
        color = WARN if "⚠️" in note else (0, 120, 0)
        d.text((560, ry+5), note, fill=color, font=get_font(12))

    # 结论
    d.line([(20, 380), (W-20, 380)], fill=(150,150,150), width=1)
    d.text((30, 390), "结论:", fill=(30,30,30), font=get_font(16))
    d.text((30, 415), "死者死于急性氰化物中毒。氰化物通过口服（普洱茶）和呼吸道（空调系统）双重途径摄入。", fill=(60,60,60), font=get_font(13))
    d.text((30, 440), "茶水中氰化钾浓度显著，为致死剂量。空调系统残留表明存在备选投毒通道。", fill=(60,60,60), font=get_font(13))

    # 签章
    d.text((W-200, H-40), "检验人: 李法医", fill=(80,80,80), font=get_font(12))
    d.text((W-200, H-20), "审核人: 张主任", fill=(80,80,80), font=get_font(12))

    path = os.path.join(OUT, "case001_toxicology.png")
    img.save(path)
    print(f"✅ {path}")
    return path


if __name__ == "__main__":
    gen_case001_scene()
    gen_case001_evidence()
    gen_monitor_screenshot()
    gen_fingerprint_comparison()
    gen_phone_screenshot()
    gen_toxicology_report()
    print("\n🎉 所有图片生成完毕!")
