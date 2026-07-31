"""Shared defaults for diagnostic plots."""

BASE_COLOR = "#356673"  # teal-slate
DIST_ALPHA = 1.0
OVERLAY_DIST_ALPHA = 0.5

METRIC_COLORS = [
    "#356673",  # teal-slate
    "#AE534C",  # terracotta
    "#6B4A6E",  # plum
    "#566B54",  # olive
]

CATEGORICAL_PALETTE = [
    "#356673",  # teal-slate
    "#AE534C",  # terracotta
    "#6B4A6E",  # plum
    "#566B54",  # olive
    "#D9A441",  # amber
    "#9C7593",  # mauve
    "#C36F63",  # dusty rose
]

METRIC_LABELS = {
    "correlation": "Correlation\n(Truth vs. Estimate)",
    "nrmse": "NRMSE",
    "contraction": "Posterior\nContraction",
    "calibration": "Calibration\nError",
}

LABEL_PAD = 10
Y_LABEL_PAD = 15
TITLE_FONTSIZE = 22
LABEL_FONTSIZE = 18
TICK_FONTSIZE = 16
HSPACE = 0.4
JOINT_HSPACE = 0.5
WSPACE = 0.2
BASE_COL_WIDTH = 6.0
BASE_ROW_HEIGHT = 3.0
