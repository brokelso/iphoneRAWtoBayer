import sys
import os
import numpy as np
import rawpy
import imageio

def extract_bayer_from_dng(filename, out_dir):
    with rawpy.imread(filename) as raw:
        raw_img = raw.raw_image_visible.astype(np.float32)
        print("raw_img.shape:", raw_img.shape)

        # Detect the raw image data layout and pick the channel/plane with most image info
        if raw_img.ndim == 2:
            print("Detected 2D raw image.")
            selected_plane = raw_img
        elif raw_img.ndim == 3:
            print("Detected 3D raw image. Inspecting axes...")
            if raw_img.shape[2] <= 8:
                print(f"Assuming last axis is channel. Found {raw_img.shape[2]} channels:")
                for c in range(raw_img.shape[2]):
                    plane = raw_img[:, :, c]
                    print(f"  Channel {c}: min={plane.min()}, max={plane.max()}, mean={plane.mean():.2f}, std={plane.std():.2f}")
                # Pick the channel with highest std
                channel_idx = np.argmax([raw_img[:, :, c].std() for c in range(raw_img.shape[2])])
                print(f"Using channel {channel_idx} (highest std) for Bayer visualization.")
                selected_plane = raw_img[:, :, channel_idx]
            else:
                print(f"Warning: shape {raw_img.shape} is unexpected; using first plane along axis 0.")
                selected_plane = raw_img[0]
        else:
            raise ValueError("Unexpected RAW image shape: " + str(raw_img.shape))

        raw_img = selected_plane
        height, width = raw_img.shape

        print("raw_img min:", np.min(raw_img), "max:", np.max(raw_img))
        minval = np.min(raw_img)
        maxval = np.max(raw_img)
        if maxval > minval:
            norm = ((raw_img - minval) / (maxval - minval) * 255).astype(np.uint8)
        else:
            norm = np.zeros_like(raw_img, dtype=np.uint8)

        # CFA pattern logic
        pattern = raw.raw_pattern
        if pattern is None:
            print("No CFA pattern found; defaulting to RGGB.")
            pattern = np.array([[0, 1], [1, 2]])
        else:
            try:
                pattern = np.array(pattern)
                if pattern.shape != (2, 2):
                    pattern = pattern.flatten()
                    if pattern.size >= 4:
                        pattern = pattern[:4].reshape((2, 2))
                        print("CFA pattern reshaped to 2x2:", pattern)
                    else:
                        pattern = np.array([[0, 1], [1, 2]])
                        print("Could not reshape CFA pattern; using RGGB.")
            except Exception:
                pattern = np.array([[0, 1], [1, 2]])
                print("Could not process CFA pattern; using RGGB.")

        print("Using CFA pattern:\n", pattern)

        # Save uninterpolated grayscale Bayer
        n = 0
        while True:
            output_fn = os.path.join(out_dir, f"dng_bayer_output{('_'+str(n)) if n else ''}.png")
            if not os.path.exists(output_fn):
                break
            n += 1
        imageio.imwrite(output_fn, norm)
        print(f"Saved uninterpolated Bayer pattern as {output_fn}")

        # Save color Bayer mosaic visualization
        color_map = [
            [255, 0, 0],    # 0: Red
            [0, 255, 0],    # 1: Green
            [0, 0, 255],    # 2: Blue
            [0, 255, 0],    # 3: Green (if present)
        ]
        color_bayer = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                cfa_idx = pattern[y % 2, x % 2]
                color = color_map[cfa_idx]
                value = int(norm[y, x])
                color_bayer[y, x, 0] = value if color[0] else 0
                color_bayer[y, x, 1] = value if color[1] else 0
                color_bayer[y, x, 2] = value if color[2] else 0

        n = 0
        while True:
            output_fn_col = os.path.join(out_dir, f"dng_bayer_color_output{('_'+str(n)) if n else ''}.png")
            if not os.path.exists(output_fn_col):
                break
            n += 1
        imageio.imwrite(output_fn_col, color_bayer)
        print(f"Saved uninterpolated color Bayer mosaic as {output_fn_col}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dng_bayer_extract.py your_image.dng")
        sys.exit(1)
    dng_path = sys.argv[1]
    out_dir = os.path.dirname(os.path.abspath(__file__))
    extract_bayer_from_dng(dng_path, out_dir)
