import os

import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon


class OutlineIoUAnalyzer:
    """
    2つの点群のIoUを計算するクラス．

    Attributes:
        source (np.ndarray | None): 点群1の座標配列．shapeは(n, 2)．
        target (np.ndarray | None): 点群2の座標配列．shapeは(n, 2)．
        iou (float | None): IoUの計算結果．
        coverage_by_target (float | None): sourceに対するtargetの網羅率．
        coverage_by_source (float | None): targetに対するsourceの網羅率．
        hull_source (ConvexHull | None): sourceの凸包．
        hull_target (ConvexHull | None): targetの凸包．
        poly_source (Polygon | None): source凸包のPolygon表現．
        poly_target (Polygon | None): target凸包のPolygon表現．

    Methods:
        __call__(source, target): 点群設定とIoU計算をまとめて実行する．
        set_points(source, target): 2つの点群を設定し，凸包を計算する．
        get_iou(): IoUと相互網羅率を返す．
        get_IoU(): get_iou()の後方互換エイリアス．
        plot_scatter(prefix="./", fig_name="IoU", **kwargs): 点群と凸包を描画して保存する．
        save_csv(prefix="./", csv_name="result.csv"): 現在の計算結果をCSV保存する．
    """

    def __init__(self):
        """
        解析状態を初期化する．
        """
        self.source = None
        self.target = None
        self.iou = None
        self.coverage_by_target = None
        self.coverage_by_source = None
        self.hull_source = None
        self.hull_target = None
        self.poly_source = None
        self.poly_target = None

    def __call__(self, source, target):
        """
        点群を設定してIoUを計算するショートカット．

        Args:
            source (np.ndarray): 点群1の座標配列．shapeは(n, 2)．
            target (np.ndarray): 点群2の座標配列．shapeは(n, 2)．

        Returns:
            dict: IoUと相互網羅率を含む辞書．
        """
        self.set_points(source, target)
        return self.get_iou()

    def set_points(self, source, target):
        """
        2つの点群を設定し，凸包（外周）を計算する．
        shape: (n, 2)
        Args:
            source np.ndarray: 点群1の座標 (n, 2)
            target np.ndarray: 点群2の座標 (n, 2)
        Returns:
            None
        """
        self.source = source
        self.target = target

        # 凸包（外周）の計算
        self.hull_source = ConvexHull(source)
        self.hull_target = ConvexHull(target)

        # 凸包（外周）の頂点を取得
        self.poly_source = Polygon(source[self.hull_source.vertices])
        self.poly_target = Polygon(target[self.hull_target.vertices])

    def get_iou(self):
        """
        2つの点群のIoUを計算する関数
        shape: (n, 2)

        Args:
            source np.ndarray: 点群1の座標 (n, 2)
            target np.ndarray: 点群2の座標 (n, 2)
        Returns:
            iou (float): IoU値
        """
        if self.poly_source is None or self.poly_target is None:
            raise ValueError("点群が設定されていません．set_points()を先に呼び出してください．")

        # IoU計算（shapelyを使用）
        intersection = self.poly_source.intersection(self.poly_target).area
        union = self.poly_source.union(self.poly_target).area
        self.iou = intersection / union if union != 0 else 0

        # 網羅率計算
        self.coverage_by_target = intersection / self.poly_source.area if self.poly_source.area != 0 else 0
        self.coverage_by_source = intersection / self.poly_target.area if self.poly_target.area != 0 else 0

        return {
            "IoU": self.iou,
            "coverage_by_target": self.coverage_by_target,
            "coverage_by_source": self.coverage_by_source,
        }

    # Backward-compatible alias for existing callers.
    def get_IoU(self):
        """
        get_iou()の後方互換エイリアス．

        Returns:
            dict: IoUと相互網羅率を含む辞書．
        """
        return self.get_iou()

    def plot_scatter(self, prefix="./", fig_name="IoU", **kwargs):
        """
        点群と凸包の散布図を描画し，画像として保存する．

        Args:
            prefix (str): 保存先ディレクトリ．
            fig_name (str): 保存ファイル名．拡張子未指定時は`.png`を付与．
            **kwargs: 描画オプション．`source_label`，`target_label`，`title`，`xlabel`，`ylabel`を指定可能．
        Returns:
            None
        """
        import matplotlib.pyplot as plt

        # 描画
        plt.figure(figsize=(8, 8))

        # 点のプロット
        source_label = kwargs.get("source_label", "Source")
        target_label = kwargs.get("target_label", "Target")
        plt.scatter(self.source[:, 0], self.source[:, 1], c="blue", label=source_label, edgecolors="w", s=80)
        plt.scatter(self.target[:, 0], self.target[:, 1], c="green", label=target_label, edgecolors="w", s=80)

        # 凸包の塗りつぶし
        plt.fill(
            self.source[self.hull_source.vertices, 0],
            self.source[self.hull_source.vertices, 1],
            "blue",
            alpha=0.2,
        )
        plt.fill(
            self.target[self.hull_target.vertices, 0],
            self.target[self.hull_target.vertices, 1],
            "green",
            alpha=0.2,
        )

        # 凸包の外周線
        for simplex in self.hull_source.simplices:
            plt.plot(self.source[simplex, 0], self.source[simplex, 1], "b-", linewidth=2)
        for simplex in self.hull_target.simplices:
            plt.plot(self.target[simplex, 0], self.target[simplex, 1], "g-", linewidth=2)

        # 表示設定
        plt.title(kwargs.get("title", f"Scatter and Outline (IoU = {self.iou:.3f})"))
        plt.xlabel(kwargs.get("xlabel", "X軸"))
        plt.ylabel(kwargs.get("ylabel", "Y軸"))
        plt.legend()
        plt.grid(True)
        plt.axis("equal")

        os.makedirs(prefix, exist_ok=True)
        save_file_path = os.path.join(prefix, fig_name)
        if not (save_file_path.endswith(".png") or save_file_path.endswith(".jpg")):
            save_file_path += ".png"

        plt.savefig(save_file_path)
        plt.close()

    def save_csv(self, prefix="./", csv_name="result.csv"):
        """
        現在の計算結果をCSVとして保存する．

        Args:
            prefix (str): 保存先ディレクトリ．
            csv_name (str): 保存ファイル名．拡張子未指定時は`.csv`を付与．

        Returns:
            dict: 保存したIoUと相互網羅率の辞書．
        """
        import pandas as pd

        result_dict = {
            "IoU": self.iou,
            "coverage_by_target": self.coverage_by_target,
            "coverage_by_source": self.coverage_by_source,
        }

        # データフレーム化
        df = pd.DataFrame([result_dict], index=None)
        # 保存するディレクトリを作成する
        os.makedirs(prefix, exist_ok=True)

        # 拡張子がなければ追加する
        if not csv_name.endswith(".csv"):
            csv_name += ".csv"

        # CSVファイルに保存する
        save_file_path = os.path.join(prefix, csv_name)
        df.to_csv(save_file_path, index=False)
        return result_dict


def _self_test():
    """
    最小限の動作確認を行う内部テスト関数．
    """
    np.random.seed(42)
    source = np.random.rand(30, 2)
    target = np.random.rand(30, 2) * 0.6 + 0.2

    analyzer = OutlineIoUAnalyzer()
    analyzer.set_points(source, target)
    result = analyzer.get_iou()

    assert isinstance(result, dict)
    assert "IoU" in result
    assert "coverage_by_target" in result
    assert "coverage_by_source" in result
    return result


if __name__ == "__main__":
    print(_self_test())
