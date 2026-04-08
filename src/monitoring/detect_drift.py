import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp, chi2_contingency

# Cấu hình logging chuyên nghiệp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

class DriftDetector:
    def __init__(self, p_val_threshold: float = 0.05, psi_threshold: float = 0.20):
        self.p_val_threshold = p_val_threshold
        self.psi_threshold = psi_threshold
        # Loại bỏ ID và Target khỏi danh sách kiểm tra drift features
        self.exclude_cols = ['CustomerID', 'Churn']

    @staticmethod
    def calculate_psi(expected, actual, bins: int = 10):
        """Tính toán Population Stability Index."""
        min_val = min(expected.min(), actual.min())
        max_val = max(expected.max(), actual.max())

        e_counts, _ = np.histogram(expected, bins=bins, range=(min_val, max_val))
        a_counts, _ = np.histogram(actual, bins=bins, range=(min_val, max_val))

        e_percents = (e_counts + 1e-6) / len(expected)
        a_percents = (a_counts + 1e-6) / len(actual)

        psi_values = (e_percents - a_percents) * np.log(e_percents / a_percents)
        return np.sum(psi_values)

    def detect_numerical(self, ref, curr, name):
        """Kiểm tra drift cho biến số (Age, Tenure, Spend...)."""
        _, p_val = ks_2samp(ref.dropna(), curr.dropna())
        psi_val = self.calculate_psi(ref.dropna(), curr.dropna())
        
        # Quyết định drift dựa trên cả ý nghĩa thống kê và độ lớn thực tế
        is_drifted = (p_val < self.p_val_threshold and psi_val > 0.1) or (psi_val >= self.psi_threshold)
        
        status = "DRIFT" if is_drifted else "STABLE"
        logger.info(f"[{status}] {name:18} | KS p-val: {p_val:.4f} | PSI: {psi_val:.4f}")
        return is_drifted

    def detect_categorical(self, ref, curr, name):
        """Kiểm tra drift cho biến phân loại (Gender, Subscription Type...)."""
        # Align categories để tránh lỗi lệch ma trận (Inhomogeneous shape)
        all_cats = sorted(set(ref.astype(str).unique()) | set(curr.astype(str).unique()))
        ref_counts = ref.astype(str).value_counts().reindex(all_cats, fill_value=0)
        curr_counts = curr.astype(str).value_counts().reindex(all_cats, fill_value=0)
        
        contingency_table = np.array([ref_counts.values, curr_counts.values])
        
        try:
            _, p_val, _, _ = chi2_contingency(contingency_table + 1e-6)
        except Exception as e:
            logger.error(f"Chi2 failed for {name}: {e}")
            p_val = 0.0

        is_drifted = p_val < self.p_val_threshold
        status = "DRIFT" if is_drifted else "STABLE"
        logger.info(f"[{status}] {name:18} | Chi2 p-val: {p_val:.4f} (Categorical)")
        return is_drifted

    def run_inference(self, train_path: Path, test_path: Path):
        try:
            df_ref = pd.read_csv(train_path)
            df_curr = pd.read_csv(test_path)
            
            logger.info("-" * 80)
            logger.info(f"STARTING DRIFT DETECTION: {train_path.name} vs {test_path.name}")
            logger.info("-" * 80)

            drifted_features = []
            
            for col in df_ref.columns:
                if col in self.exclude_cols:
                    continue
                
                # Logic phân loại kiểu dữ liệu dựa trên thực tế DataFrame của bạn
                if pd.api.types.is_numeric_dtype(df_ref[col]) and df_ref[col].nunique() > 10:
                    if self.detect_numerical(df_ref[col], df_curr[col], col):
                        drifted_features.append(col)
                else:
                    if self.detect_categorical(df_ref[col], df_curr[col], col):
                        drifted_features.append(col)

            return drifted_features

        except Exception as e:
            logger.error(f"Process failed: {str(e)}")
            raise

if __name__ == "__main__":
    # Cấu hình đường dẫn
    ROOT = Path(__file__).resolve().parents[2]
    TRAIN_CSV = ROOT / "data" / "raw" / "train.csv"
    TEST_CSV = ROOT / "data" / "raw" / "test.csv"

    detector = DriftDetector(p_val_threshold=0.05, psi_threshold=0.20)
    drifted_cols = detector.run_inference(TRAIN_CSV, TEST_CSV)

    print("-" * 80)
    if drifted_cols:
        logger.warning(f"RESULT: {len(drifted_cols)} features drifted: {drifted_cols}")
        # sys.exit(1) # Bật cái này nếu muốn dừng CI/CD pipeline
    else:
        logger.info("RESULT: All features are stable. Model is safe to deploy.")