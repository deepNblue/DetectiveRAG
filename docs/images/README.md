# 📸 系统截图

请将 WebUI 启动截图保存为 `webui_screenshot.png`

## 如何截图

1. 访问 WebUI: http://localhost:7860
2. 运行一个案件分析
3. 截取包含以下内容的界面：
   - 案件输入区
   - 专家群聊面板
   - 证据图谱可视化
4. 保存为 `docs/images/webui_screenshot.png`

## Linux 截图命令

```bash
# 使用 gnome-screenshot
gnome-screenshot -a  # 区域选择截图

# 使用 import (ImageMagick)
import docs/images/webui_screenshot.png

# 从剪贴板保存（如果已截图）
xclip -selection clipboard -t image/png -o > docs/images/webui_screenshot.png
```

## macOS 截图命令

```bash
# 截图到剪贴板: Cmd+Shift+4
# 从剪贴板保存
pngpaste docs/images/webui_screenshot.png
```
