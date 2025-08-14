import argparse
import cv2
import numpy as np

def _ensure_float(img):
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    else:
        return img.astype(np.float32)

def _to_uint8(img):
    img = np.clip(img, 0, 1)
    return (img * 255.0 + 0.5).astype(np.uint8)

def single_scale_retinex(channel, sigma):
    """Retinex on one channel at one Gaussian scale."""
    # Add small epsilon to avoid log(0)
    eps = 1e-6
    blur = cv2.GaussianBlur(channel, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    return np.log(channel + eps) - np.log(blur + eps)

def multi_scale_retinex(img, sigmas=(15, 80, 250), weights=None):
    """
    Apply MSR per channel on an RGB [0..1] image.
    sigmas: tuple/list of Gaussian sigmas (pixels)
    weights: same length as sigmas, default equal weights
    """
    if weights is None:
        weights = [1.0 / len(sigmas)] * len(sigmas)
    msr = np.zeros_like(img, dtype=np.float32)
    for w, s in zip(weights, sigmas):
        for c in range(3):
            msr[:, :, c] += w * single_scale_retinex(img[:, :, c], s)
    return msr

def color_restoration(img, alpha=125.0, beta=46.0):
    """
    MSRCR color restoration term (Land/Jobson-style).
    img is RGB [0..1].
    alpha and beta control strength and slope.
    """
    eps = 1e-6
    # per-pixel sum across channels
    I_sum = np.sum(img, axis=2, keepdims=True) + eps
    # C(x) = beta * [ log(alpha * I_c(x)) - log( I_sum(x) ) ]
    C = beta * (np.log(alpha * img + eps) - np.log(I_sum))
    return C

def simplest_color_balance(img, s1=0.01, s2=0.01):
    """
    Clip low/high percentiles and rescale. Works channel-wise on [0..1] RGB.
    s1/s2 are percentages (0..1). Default 1%/1%.
    """
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        ch = img[:, :, c]
        lo = np.percentile(ch, s1 * 100.0)
        hi = np.percentile(ch, 100.0 - s2 * 100.0)
        if hi - lo < 1e-6:
            out[:, :, c] = ch
        else:
            ch = np.clip((ch - lo) / (hi - lo), 0, 1)
            out[:, :, c] = ch
    return out

def msrcr(img_bgr,
          sigmas=(15, 80, 250),
          weights=None,
          alpha=125.0,
          beta=46.0,
          gain=1.0,
          offset=0.0,
          do_color_balance=True,
          balance_low=0.01,
          balance_high=0.01,
          out_gamma=1.0):
    """
    Full MSRCR pipeline.
    - img_bgr: uint8/uint16/float BGR image
    - out_gamma: optional gamma after normalization (e.g., 0.9–1.2)
    """
    # Convert to RGB [0..1]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = _ensure_float(img_rgb)

    # Multi-Scale Retinex
    msr = multi_scale_retinex(img, sigmas=sigmas, weights=weights)

    # Color Restoration and gain/offset
    C = color_restoration(img, alpha=alpha, beta=beta)
    out = gain * (msr * C) + offset  # combine

    # Normalize to [0..1] per-channel using min-max on the Retinex result
    out -= out.min(axis=(0, 1), keepdims=True)
    denom = out.max(axis=(0, 1), keepdims=True)
    denom[denom < 1e-6] = 1.0
    out = out / denom

    # Optional color balance to remove residual color casts / clipping
    if do_color_balance:
        out = simplest_color_balance(out, s1=balance_low, s2=balance_high)

    # Optional gamma for brightness feel
    if out_gamma != 1.0:
        out = np.power(np.clip(out, 0, 1), 1.0 / out_gamma)

    # Back to BGR uint8
    out_bgr = cv2.cvtColor(_to_uint8(out), cv2.COLOR_RGB2BGR)
    return out_bgr

def msrcr_lowlight_preset(img_bgr):
    """
    A sensible preset for very low-light handheld photos:
    - Mix of small/medium/large sigmas
    - Slightly stronger color balance
    - Mild brightening gamma
    """
    return msrcr(
        img_bgr,
        sigmas=(15, 60, 200),
        weights=None,         # equal weights
        alpha=125.0,
        beta=46.0,
        gain=1.0,
        offset=0.0,
        do_color_balance=True,
        balance_low=0.01,
        balance_high=0.01,
        out_gamma=0.9         # <1 brightens midtones a bit
    )

def optional_final_touch(img_bgr):
    """
    Optional: Light denoise + CLAHE in L*a*b* to lift shadows without killing color.
    Keep gentle to avoid artifacts.
    """
    # Light denoise
    den = cv2.fastNlMeansDenoisingColored(img_bgr, None, 5, 5, 7, 21)

    # CLAHE on L channel
    lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    lab2 = cv2.merge((L2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def main():
    parser = argparse.ArgumentParser(description="Low-light enhancement via MSRCR (Retinex).")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to save enhanced image")
    parser.add_argument("--preset", action="store_true", help="Use low-light preset (recommended)")
    parser.add_argument("--final_touch", action="store_true", help="Apply optional denoise + CLAHE")
    args = parser.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.input}")

    if args.preset:
        enhanced = msrcr_lowlight_preset(img)
    else:
        # Default general-purpose parameters
        enhanced = msrcr(
            img,
            sigmas=(15, 80, 250),
            weights=None,
            alpha=125.0,
            beta=46.0,
            gain=1.0,
            offset=0.0,
            do_color_balance=True,
            balance_low=0.01,
            balance_high=0.01,
            out_gamma=1.0
        )

    if args.final_touch:
        enhanced = optional_final_touch(enhanced)

    cv2.imwrite(args.output, enhanced)
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
