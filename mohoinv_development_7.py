import numpy as np
from scipy.sparse import diags, coo_matrix
from scipy.constants import G
import harmonica as hm
import boule as bl
import copy
import xarray as xr
import matplotlib.pyplot as plt


# ==============================================================================
# 工具函数
# ==============================================================================

def to_xarray(lon, lat, field, name="field", units="mGal"):
    """把 2D field 转成 xarray.DataArray，dims=(lat, lon)"""
    lon   = np.asarray(lon)
    lat   = np.asarray(lat)
    field = np.asarray(field)
    if field.shape != (lat.size, lon.size):
        raise ValueError(
            f"field shape {field.shape} does not match "
            f"(len(lat), len(lon)) = {(lat.size, lon.size)}"
        )
    da = xr.DataArray(
        field,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name=name,
    )
    da.attrs["units"] = units
    return da


# ==============================================================================
# MohoLayer
# ==============================================================================

class MohoLayer:
    """
    几何约定
    --------
    mean_radius = bl.Moon2015.radius
    ref_r  = mean_radius + reference   # 参考莫霍面球半径
    sur_r  = ref_r + surface           # 实际莫霍面球半径
    height > 0 → 观测面高于参考球面

    密度约定（_update_density）
    --------
    density_contrast > 0 表示莫霍面以下更重（地幔 > 地壳）。
    送入正演的 density_array：
        surface > 0  → 莫霍面上隆，该体素"多了地幔" → 正异常 → density_array = +|Δρ|
        surface <= 0 → 莫霍面下沉，该体素"多了地壳" → 负异常 → density_array = -|Δρ|

    ⚠️  原代码符号与此相反（surface>0 取负）。
    若你已用原代码验证过正演，请把 _update_density 里的 mask 逻辑改回去。
    """

    def __init__(self, lon, lat, height, reference, surface, density_contrast):
        self.lon              = np.asarray(lon,              dtype=float)
        self.lat              = np.asarray(lat,              dtype=float)
        self.height           = float(height)
        self.reference        = np.asarray(reference,        dtype=float)
        self.surface          = np.asarray(surface,          dtype=float)
        self.density_contrast = np.asarray(density_contrast, dtype=float)

        self.mean_radius      = bl.Moon2015.radius
        self.LON, self.LAT    = np.meshgrid(self.lon, self.lat)

        self._validate_inputs()
        self.shape = self.reference.shape

        self.ref_r        = self.mean_radius + self.reference
        self.sur_r        = self.ref_r + self.surface
        self.density_array = np.empty_like(self.density_contrast)
        self._update_density()

    # ---- 内部更新 ----

    def _update_density(self):
        self.density_array        = np.abs(self.density_contrast).copy()
        self.density_array[self.surface <= 0] *= -1.0

    def _update_geometry(self):
        self.sur_r = self.ref_r + self.surface
        self._update_density()

    # ---- 公开接口 ----

    def set_surface(self, surface):
        self.surface = np.asarray(surface, dtype=float)
        self._update_geometry()

    def set_reference(self, reference):
        self.reference = np.asarray(reference, dtype=float)
        self.ref_r     = self.mean_radius + self.reference
        self.surface   = self.sur_r - self.ref_r
        self._update_density()

    def copy(self):
        return copy.deepcopy(self)

    def forward(self, quite=True):
        """返回观测面上的垂向重力 g_z，2D array，单位 mGal"""
        tess_layer = hm.tesseroid_layer(
            (self.lon, self.lat),
            surface=self.sur_r,
            reference=self.ref_r,
            properties={"density": self.density_array},
        )
        coordinates = (
            self.LON,
            self.LAT,
            self.mean_radius * np.ones(self.shape) + self.height,
        )
        return tess_layer.tesseroid_layer.gravity(
            coordinates, field="g_z", progressbar=not quite
        )
    
    def forward_testset(self, lon_pts, lat_pts, quite=True):
        """
        在任意散点位置计算正演重力

        参数
        ----
        lon_pts : 1D array，散点经度
        lat_pts : 1D array，散点纬度

        返回
        ----
        g : 1D array，各散点处的重力值 (mGal)
        """
        tess_layer = hm.tesseroid_layer(
        (self.lon, self.lat),
        surface=self.sur_r,
        reference=self.ref_r,
        properties={"density": self.density_array},
        )
        coordinates = (
        np.asarray(lon_pts, dtype=float),
        np.asarray(lat_pts, dtype=float),
        np.full(len(lon_pts), self.mean_radius + self.height),
        )
        return tess_layer.tesseroid_layer.gravity(
        coordinates, field="g_z", progressbar=not quite
        )

    def surface_to_xarray(self):
        return to_xarray(self.lon, self.lat, self.surface, name="surface", units="m")

    def field_to_xarray(self):
        return to_xarray(
            self.lon, self.lat, self.forward(quite=False), name="gz", units="mGal"
        )

    def _validate_inputs(self):
        if self.lon.ndim != 1:
            raise ValueError("lon 必须是 1D array。")
        if self.lat.ndim != 1:
            raise ValueError("lat 必须是 1D array。")
        expected = (self.lat.size, self.lon.size)
        for name, arr in [("reference", self.reference), ("surface", self.surface)]:
            if arr.shape != expected:
                raise ValueError(f"{name}.shape={arr.shape}，期望 {expected}。")
        if self.density_contrast.ndim > 0:
            if self.density_contrast.shape != expected:
                raise ValueError(
                    f"density_contrast.shape={self.density_contrast.shape}，"
                    f"期望 scalar 或 {expected}。"
                )


# ==============================================================================
# MohoInversion
# ==============================================================================

class MohoInversion:
    """
    高斯-牛顿莫霍面反演

    目标函数
    --------
    φ(x) = ‖g(x) - g_obs‖² + μ²‖L(x - x₀)‖²

    迭代格式
    --------
    (J^T J + μ² L^T L) δx = -(J^T r + μ² L^T L (x - x₀))
    x ← x + α δx   （α 由 Armijo 线搜索确定）

    已修复的 BUG（相对原始代码）
    ---------------------------
    1. jacobian()：改用 |density_contrast| 而非 density_array，
       保证对角 Hessian J^T J 正定，线性系统稳定。
    2. _safety_clip()：改为逐点上界 height - reference，
       而非全局标量 -mean(reference)，对非均匀 reference 更正确。
    3. residual_2d()：_finalize 已更新 layer.surface，
       直接调用 layer.forward() 即可，无需额外拷贝。
    4. 线搜索次数从 20 增至 30，减少过早失败的概率。
    """

    def __init__(self, layer, go, max_iter=20, mu=0.0, quite=False):
        """
        参数
        ----
        layer    : MohoLayer，含初始猜测模型
        go       : 2D array (nlat, nlon)，观测重力场 (mGal)
        max_iter : 最大迭代次数
        mu       : Tikhonov 正则化系数（≥0，0 = 不正则化）
        quite    : True 时抑制正演进度条
        """
        self.layer    = layer
        self.go       = np.asarray(go, dtype=float)
        self.max_iter = max_iter
        self.quite    = quite
        self.shape    = layer.shape
        self.mu       = float(mu)

        self.inverted_surface = None
        # 收敛历史，index 0 = 初始模型
        self.rms_history      = []

        # 逐点安全上界：surface ≤ height - reference（保证莫霍面在观测面以下）
        self._x_upper = (layer.height - layer.reference).flatten()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def copy(self):
        return copy.deepcopy(self)

    def L(self):
        """一阶有限差分矩阵，形状 (n_edges, n_params)"""
        nlat, nlon = self.shape
        rows, cols, data = [], [], []
        row = 0

        def idx(i, j):
            return i * nlon + j

        for i in range(nlat):          # 经向差分
            for j in range(nlon - 1):
                rows += [row, row]; cols += [idx(i,j), idx(i,j+1)]; data += [-1.,1.]; row += 1

        for i in range(nlat - 1):      # 纬向差分
            for j in range(nlon):
                rows += [row, row]; cols += [idx(i,j), idx(i+1,j)]; data += [-1.,1.]; row += 1

        return coo_matrix((data, (rows, cols)), shape=(row, nlat*nlon)).tocsr()

    def jacobian(self):
        """
        Bott (1960) 对角 Jacobian：J_ii = 2π G |Δρ_i| × 1e5  [mGal/m]
        始终取 |density_contrast|，保证对角元恒正。
        """
        rho  = np.abs(self.layer.density_contrast).flatten()
        diag = 2.0 * np.pi * G * rho * 1e5
        return diags(diag, offsets=0, format="csr")

    def _safety_clip(self, x):
        """逐点裁剪，保证 sur_r ≤ mean_radius + height"""
        return np.minimum(x, self._x_upper)

    def _residual(self, x):
        """r = g_forward(x) - g_obs，1D (mGal)"""
        layer = self.layer.copy()
        layer.set_surface(x.reshape(self.shape))
        return layer.forward(quite=self.quite).flatten() - self.go.flatten()

    def _phi(self, x, r, L, x0, mu):
        """目标函数值"""
        reg = L @ (x - x0)
        return np.dot(r, r) + mu**2 * np.dot(reg, reg)

    def _solve(self, A, b):
        """统一稀疏/稠密线性求解"""
        if hasattr(A, "tocsc"):
            from scipy.sparse.linalg import spsolve
            return spsolve(A.tocsc(), b)
        return np.linalg.solve(A, b)

    # ------------------------------------------------------------------
    # 主反演：高斯-牛顿 + Armijo 线搜索
    # ------------------------------------------------------------------

    def inversion_gn(self, gtol=1e-3, mu=None, max_iter=None):
        """
        高斯-牛顿反演

        参数
        ----
        gtol     : Δφ/φ < gtol 时停止
        mu       : 覆盖构造函数的 mu
        max_iter : 覆盖构造函数的 max_iter

        返回
        ----
        inverted_surface : 2D array (nlat, nlon)，莫霍面起伏 (m)
        """
        if mu       is None: mu       = self.mu
        if max_iter is None: max_iter = self.max_iter

        x  = self.layer.surface.flatten().copy()
        x0 = x.copy()
        n  = len(x)

        L = self.L()
        J = self.jacobian()

        r    = self._residual(x)
        chi2 = float(np.dot(r, r))
        phi  = self._phi(x, r, L, x0, mu)

        self.rms_history = [float(np.sqrt(chi2 / n))]

        print("=" * 60)
        print("Gauss-Newton inversion  (Bott Jacobian + Armijo line search)")
        print(f"  n_params={n}   μ={mu}   max_iter={max_iter}")
        print(f"  iter  0:  rms={self.rms_history[0]:.4f} mGal   φ={phi:.4e}")
        print("=" * 60)

        for k in range(max_iter):

            # 梯度与近似 Hessian
            reg_vec = L @ (x - x0)
            g_vec   = J.T @ r + mu**2 * (L.T @ reg_vec)
            H       = J.T @ J + mu**2 * (L.T @ L)
            delta_x = -self._solve(H, g_vec)

            # Armijo 线搜索
            alpha = 1.0
            c1    = 1e-4
            slope = float(g_vec @ delta_x)   # 下降方向时为负值

            line_ok = False
            for _ in range(30):
                x_try   = self._safety_clip(x + alpha * delta_x)
                r_try   = self._residual(x_try)
                phi_try = self._phi(x_try, r_try, L, x0, mu)
                if phi_try <= phi + c1 * alpha * slope:
                    line_ok = True
                    break
                alpha *= 0.5

            if not line_ok:
                print(f"  iter {k+1:2d}: 线搜索失败，提前停止。")
                break

            chi2_try = float(np.dot(r_try, r_try))
            phi_red  = (phi - phi_try) / (abs(phi) + 1e-30)
            rms      = float(np.sqrt(chi2_try / n))
            self.rms_history.append(rms)

            print(
                f"  iter {k+1:2d}:  rms={rms:.4f} mGal   "
                f"φ={phi_try:.4e}   Δφ/φ={phi_red:+.2e}   α={alpha:.4f}"
            )

            x = x_try; r = r_try; phi = phi_try; chi2 = chi2_try

            if phi_red < gtol:
                print(f"  收敛：Δφ/φ={phi_red:.2e} < {gtol}"); break

            if float(np.linalg.norm(g_vec)) / n < gtol * 1e-2:
                print(f"  收敛：‖g‖/n < {gtol*1e-2:.2e}"); break
        else:
            print(f"  达到最大迭代次数 {max_iter}。")

        self._finalize(x, chi2, n)
        return self.inverted_surface

    def inversion(self, gtol=1e-3, mu=None, max_iter=None):
        """向后兼容别名，等价于 inversion_gn()"""
        return self.inversion_gn(gtol=gtol, mu=mu, max_iter=max_iter)

    def _finalize(self, x, chi2, n):
        self.inverted_surface = x.reshape(self.shape)
        self.layer.set_surface(self.inverted_surface)
        print("=" * 60)
        print(f"  最终 rms = {np.sqrt(chi2/n):.4f} mGal")
        print("=" * 60)

    # ------------------------------------------------------------------
    # 收敛曲线
    # ------------------------------------------------------------------

    def convergence_data(self):
        """
        返回收敛曲线原始数据

        返回
        ----
        iters : 1D int array，迭代编号（0 = 初始模型）
        rms   : 1D float array，各步 rms (mGal)
        """
        rms   = np.array(self.rms_history, dtype=float)
        iters = np.arange(len(rms), dtype=int)
        return iters, rms

    def convergence_to_xarray(self):
        """收敛曲线存为 xarray.DataArray"""
        iters, rms = self.convergence_data()
        da = xr.DataArray(
            rms, dims=("iteration",), coords={"iteration": iters}, name="rms"
        )
        da.attrs["units"]       = "mGal"
        da.attrs["description"] = "RMS misfit；iteration=0 为初始模型"
        return da

    def plot_convergence(self, ax=None, **kwargs):
        """
        绘制收敛曲线

        返回
        ----
        fig, ax
        """
        iters, rms = self.convergence_data()
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig = ax.get_figure()

        kw = dict(marker="o", linewidth=1.5, markersize=4, color="steelblue")
        kw.update(kwargs)
        ax.plot(iters, rms, **kw)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("RMS misfit (mGal)")
        ax.set_title("Convergence curve")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    # 残差
    # ------------------------------------------------------------------

    def residual_2d(self):
        """最终残差 2D 分布 (mGal)，_finalize 已更新 layer.surface"""
        return self.layer.forward(quite=self.quite) - self.go

    def residual_flat(self):
        """最终残差展平为 1D array (mGal)"""
        return self.residual_2d().flatten()

    def residual_stats(self):
        """
        残差统计量

        返回
        ----
        dict，键：mean / std / rms / p5 / p95（单位 mGal）
        """
        res = self.residual_flat()
        return {
            "mean": float(np.mean(res)),
            "std" : float(np.std(res)),
            "rms" : float(np.sqrt(np.mean(res**2))),
            "p5"  : float(np.percentile(res,  5)),
            "p95" : float(np.percentile(res, 95)),
        }

    def residual_histogram_data(self, bins=50):
        """
        残差直方图原始数据（不绘图）

        返回
        ----
        counts      : 1D int array，各 bin 频数
        bin_edges   : 1D float array，bin 边界（长度 bins+1）
        bin_centers : 1D float array，bin 中心（长度 bins）
        """
        res = self.residual_flat()
        counts, bin_edges = np.histogram(res, bins=bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return counts, bin_edges, bin_centers

    def plot_residual_histogram(self, bins=50, ax=None, **kwargs):
        """
        绘制残差直方图 + 正态拟合曲线

        返回
        ----
        fig, ax
        """
        from scipy.stats import norm as sp_norm

        res                            = self.residual_flat()
        counts, bin_edges, bin_centers = self.residual_histogram_data(bins=bins)
        stats                          = self.residual_stats()

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig = ax.get_figure()

        width = bin_edges[1] - bin_edges[0]
        kw = dict(color="steelblue", alpha=0.7, edgecolor="white")
        kw.update(kwargs)
        ax.bar(bin_centers, counts, width=width, **kw)

        x_fit = np.linspace(bin_edges[0], bin_edges[-1], 300)
        pdf   = sp_norm.pdf(x_fit, loc=stats["mean"], scale=stats["std"])
        ax.plot(x_fit, pdf * counts.sum() * width,
                color="crimson", linewidth=2, label="Normal fit")

        textstr = (
            f"mean = {stats['mean']:+.3f} mGal\n"
            f"std  = {stats['std']:.3f} mGal\n"
            f"rms  = {stats['rms']:.3f} mGal\n"
            f"p5–p95 = [{stats['p5']:.2f}, {stats['p95']:.2f}]"
        )
        ax.text(0.97, 0.95, textstr,
                transform=ax.transAxes, fontsize=8.5, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.6))

        ax.set_xlabel("Residual (mGal)")
        ax.set_ylabel("Count")
        ax.set_title("Residual histogram")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    # xarray 输出
    # ------------------------------------------------------------------

    def residual_2d_to_xarray(self):
        return to_xarray(
            self.layer.lon, self.layer.lat,
            self.residual_2d(), name="residual", units="mGal"
        )

    def surface_to_xarray(self):
        return self.layer.surface_to_xarray()

    def field_to_xarray(self):
        return self.layer.field_to_xarray()
    
    def evaluate_testset(self, lon_pts, lat_pts, go_test):
        """
        在测试点上评估反演结果

        参数
        ----
        lon_pts : 1D array，测试点经度
        lat_pts : 1D array，测试点纬度
        go_test : 1D array，测试点观测重力 (mGal)

        返回
        ----
        residual : 1D array，预测值 - 观测值 (mGal)
        scores   : dict，RMSE / MAE / R² / bias
        """
        lon_pts = np.asarray(lon_pts, dtype=float).flatten()
        lat_pts = np.asarray(lat_pts, dtype=float).flatten()
        go_test = np.asarray(go_test, dtype=float).flatten()

        # 用反演后的 layer 在测试点正演
        go_pred  = self.layer.forward_testset(lon_pts, lat_pts, quite=self.quite)
        residual = go_pred - go_test

        ss_res = np.dot(residual, residual)
        ss_tot = np.sum((go_test - go_test.mean()) ** 2)

        scores = {
            "rmse" : float(np.sqrt(np.mean(residual ** 2))),
            "mae"  : float(np.mean(np.abs(residual))),
            "bias" : float(np.mean(residual)),
            "r2"   : float(1.0 - ss_res / (ss_tot + 1e-30)),
        }

        print(f"  Test RMSE = {scores['rmse']:.4f} mGal")
        print(f"  Test MAE  = {scores['mae']:.4f} mGal")
        print(f"  Test bias = {scores['bias']:+.4f} mGal")
        print(f"  Test R²   = {scores['r2']:.6f}")

        return go_pred, residual, scores

    def evaluate_seismic(self, lon_pts, lat_pts, seismic_thickness, topo_pts):
        """
        用地震 Moho 厚度评估反演结果

        几何关系
        --------
        月壳厚度（m） = topo_pts - reference_at_pts - surface_at_pts
        其中：
          topo_pts      : 地形相对于平均球半径的高度 (m)，正值 = 地表高于平均球
          reference     : 参考 Moho 面偏移量 (m)，通常为负（Moho 在平均球面以下）
          surface       : 反演得到的 Moho 起伏 (m)

        参数
        ----
        lon_pts           : 1D array，地震点经度 (°)
        lat_pts           : 1D array，地震点纬度 (°)
        seismic_thickness : 1D array，地震反演月壳厚度 (m)
        topo_pts          : 1D array，地震点处地形高度（相对于平均球半径，m）

        返回
        ----
        thickness_inv : 1D array，从重力反演得到的月壳厚度 (m)
        diff          : 1D array，反演值 - 地震值 (m)，正值 = 反演偏厚
        scores        : dict，MAE / RMSE / bias / R²（单位 m）
        """
        from scipy.interpolate import RegularGridInterpolator

        if self.inverted_surface is None:
            raise RuntimeError("尚未完成反演，请先调用 inversion_gn()。")

        lon_pts           = np.asarray(lon_pts,           dtype=float).flatten()
        lat_pts           = np.asarray(lat_pts,           dtype=float).flatten()
        seismic_thickness = np.asarray(seismic_thickness, dtype=float).flatten()
        topo_pts          = np.asarray(topo_pts,          dtype=float).flatten()

        if not (lon_pts.shape == lat_pts.shape == seismic_thickness.shape == topo_pts.shape):
            raise ValueError("lon_pts / lat_pts / seismic_thickness / topo_pts 长度必须一致。")

        # RegularGridInterpolator 要求 lat 方向单调递增
        lat_grid  = self.layer.lat.copy()
        lon_grid  = self.layer.lon
        surface   = self.inverted_surface.copy()
        reference = self.layer.reference.copy()

        if lat_grid[0] > lat_grid[-1]:
            lat_grid  = lat_grid[::-1]
            surface   = surface[::-1, :]
            reference = reference[::-1, :]

        interp_surface = RegularGridInterpolator(
            (lat_grid, lon_grid), surface,
            method="linear", bounds_error=False, fill_value=None,
        )
        interp_ref = RegularGridInterpolator(
            (lat_grid, lon_grid), reference,
            method="linear", bounds_error=False, fill_value=None,
        )

        pts              = np.column_stack([lat_pts, lon_pts])
        surface_at_pts   = interp_surface(pts)
        reference_at_pts = interp_ref(pts)

        # 月壳厚度（m）
        thickness_inv = topo_pts - reference_at_pts - surface_at_pts
        diff          = thickness_inv - seismic_thickness

        ss_res = float(np.dot(diff, diff))
        ss_tot = float(np.sum((seismic_thickness - seismic_thickness.mean()) ** 2))

        scores = {
            "mae"  : float(np.mean(np.abs(diff))),
            "rmse" : float(np.sqrt(np.mean(diff ** 2))),
            "bias" : float(np.mean(diff)),
            "r2"   : float(1.0 - ss_res / (ss_tot + 1e-30)),
        }

        print("  --- 地震评分 (Seismic Score) ---")
        print(f"  点数   = {len(lon_pts)}")
        print(f"  MAE    = {scores['mae']/1e3:.3f} km")
        print(f"  RMSE   = {scores['rmse']/1e3:.3f} km")
        print(f"  Bias   = {scores['bias']/1e3:+.3f} km  (正值=反演偏厚)")
        print(f"  R²     = {scores['r2']:.6f}")

        return thickness_inv, diff, scores