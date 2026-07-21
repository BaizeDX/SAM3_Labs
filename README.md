# SAM 3 - 交互式图像分割标注工具

基于 **Meta SAM 3 (Segment Anything Model 3)** 的交互式图像分割标注前端+后端系统，支持四种标注模式、多边形人工细抠、负框排除、别名标注等功能。

## 项目结构

```
test/
├── backend/
│   ├── main.py              # FastAPI 后端（SAM 3 推理服务器）
│   └── requirements.txt     # Python 依赖
├── frontend/
│   └── index.html           # 单页 Web UI（所有交互逻辑）
└── start_server.bat         # Windows 启动脚本
config/
├── batch_prompts.txt        # A 模式预制提示词
└── batch_prompts_bc.txt     # B/C/D 模式预制提示词
Inputs/RawImages/            # 输入图片目录
Outputs/LabABC_output/       # 导出结果目录
```

## 功能总览

### 四种标注模式

| 模式 | 名称 | 操作 | 适用场景 |
|------|------|------|---------|
| **A** 🟦 | 纯文本分割 | 选择/输入文本 → 点击 Run | 简单结构体（墙面、屋顶、路面） |
| **B** 🟩 | 框+文本多掩码 | 左键拖框 + 输入文本 → 点击 Run | 复杂组件（多个同类部件候选） |
| **C** 🟥 | 框+文本单掩码 | 左键拖框 + 输入文本 → 点击 Run | 精确定位（只取最高分掩码，裁剪到框范围） |
| **D** 🟨 | 手动多边形 | 点击画布添加顶点 → 点击起点闭合 | 人工细抠（模型无法分割的复杂区域） |

### 负框排除

在 B/C 模式下，**右键拖拽**可以绘制蓝色虚线负框，通知模型排除该区域的掩码像素。支持添加多个负框，结果叠加应用。

- 右键拖拽 → 蓝色虚线负框（带 `−` 标签）
- 松开确认（最小 5px 避免误触）
- 🗑 负框 按钮一键清除所有负框

### 别名系统（B/C/D 模式）

可在工具栏中分别填写 **Text（搜索词）** 和 **Alias（标注名）**：

| 搜索词 (Text) | 别名 (Alias) | ☑ 启用 | 实例 ID |
|-------------|-------------|--------|---------|
| `屋檐` | 留空 | — | `D_屋檐_#01` |
| `屋檐` | `燕尾檐` | ☑ | `D_燕尾檐_#01` |
| `屋檐` | `燕尾檐` | ☐ | `D_屋檐_#01` |

### 多边形标注（D 模式）

- **左键单击**：添加多边形顶点
- **拖拽顶点**：调整已添加的顶点位置
- **点击起点（红色顶点 1）**：闭合多边形并生成掩码
- **↩ 撤销点**：回退上一个顶点
- **🗑 清除**：清除当前多边形
- **✓ 创建蒙版**：手动闭合多边形并创建掩码
- **✏️（掩码列表）**：重新编辑已有 D 掩码的多边形

### 放大查看

- 点击 Full Size 按钮打开全尺寸放大视图
- **滚轮缩放**·**拖拽平移**
- **ESC** 关闭·**+/-** 缩放·**0** 复位
- 可切换标签显示

### 重叠检测

每次创建新掩码后自动检测与已有掩码的重叠：

| IoU 阈值 | 显示 |
|---------|------|
| ≥ 10% | 🟡 状态栏警告 |
| ≥ 30% | 🔴 状态栏警告 |
| — | 掩码列表中标红并提示 `⚠️ 与 X, Y 重叠` |

检测包含已确认（hidden）的掩码，避免重复标注。

### 掩码管理

- 复选框切换掩码可见性
- 按模式/分数排序
- Highlight 高亮选中掩码
- 评估标记（Under/Over Segmentation、Boundary）
- Push All Pending → 推送到已确认
- 导出为 PNG + JSON 元数据

## 使用技术

### 后端

- **[SAM 3](https://ai.meta.com/sam3)** by Meta：核心分割模型，从 `sam3.pt` 检查点加载
- **FastAPI**：RESTful API 服务器
- **PyTorch + CUDA**：GPU 推理（自动检测 CUDA，回退 CPU）
- **Pillow / NumPy**：图像处理与掩码操作

### API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/images` | GET | 列出所有图片 |
| `/api/image/{id}` | GET | 获取单张图片 + 尺寸 |
| `/api/config-prompts` | GET | A 模式提示词列表 |
| `/api/config-prompts-bc` | GET | B/C 模式提示词列表 |
| `/api/segment-a` | POST | A 模式分割 |
| `/api/segment-b` | POST | B 模式分割（支持 `negative_boxes`） |
| `/api/segment-c` | POST | C 模式分割（支持 `negative_boxes`） |
| `/api/export` | POST | 导出掩码结果 |

### 并发安全

后端使用 `asyncio.Lock` 串行化 GPU 推理访问，杜绝多人同时使用时的竞态条件。

## 快速开始

### 环境要求

- Python 3.12+
- PyTorch 2.7+ (CUDA 12.6+)
- CUDA 兼容 GPU（可选，CPU 也可用）

### 安装

```bash
# 1. 创建环境
conda create -n sam3 python=3.12
conda activate sam3

# 2. 安装 PyTorch
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. 安装 SAM 3
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .

# 4. 安装本项目依赖
cd test/backend
pip install -r requirements.txt

# 5. 下载 SAM 3 检查点
# 从 https://huggingface.co/facebook/sam3 下载 sam3.pt
# 放置到项目根目录
```

### 运行

```bash
cd test/backend
python main.py
```

浏览器访问 `http://localhost:8501` 即可使用。

### 使用内网穿透分享

```bash
# 安装 ngrok 后
ngrok http 8501
```

将生成的公网 URL 分享给其他人，即可协作标注。

## 配置

### 提示词配置

- `config/batch_prompts.txt` — A 模式的默认提示词列表（每行一个）
- `config/batch_prompts_bc.txt` — B/C/D 模式的默认提示词列表（每行一个）

### 后端启动参数

后端默认扫描：
- 模型文件：`{repo_root}/sam3.pt`
- 输入图片：`{repo_root}/Inputs/RawImages/`
- 输出目录：`{repo_root}/Outputs/LabABC_output/`

## 许可证

本项目基于 SAM 3 模型，遵循其原始许可证。详见 [`LICENSE`](LICENSE)。
