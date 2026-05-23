# outline-iou-analyzer

2つの点群から凸包を作成し，IoUと相互網羅率を計算するPythonライブラリです．

## Installation

### PyPI以外（GitHubから直接）

```bash
pip install "git+https://github.com/akh1r0ck/outline-iou-analyzer.git"
```

### ブランチやタグを指定する場合

```bash
pip install "git+https://github.com/akh1r0ck/outline-iou-analyzer.git@main"
pip install "git+https://github.com/akh1r0ck/outline-iou-analyzer.git@v0.1.0"
```

## Usage

```python
import numpy as np
from outline_iou_analyzer import OutlineIoUAnalyzer

source = np.random.rand(30, 2)
target = np.random.rand(30, 2) * 0.6 + 0.2

analyzer = OutlineIoUAnalyzer()
analyzer.set_points(source, target)
result = analyzer.get_iou()

print(result["IoU"])
print(result["coverage_by_target"])
print(result["coverage_by_source"])
```

## License

MIT License
