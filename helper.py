import cv2
import numpy as np

def apply_proj(corners, f, tilt_x, tilt_y, offset_x, offset_y, dst_width, dst_height, scale):
  # Rotation matrices
  Rx = np.array([
    [1, 0, 0],
    [0, np.cos(tilt_x), -np.sin(tilt_x)],
    [0, np.sin(tilt_x),  np.cos(tilt_x)]
  ])
  Ry = np.array([
    [np.cos(tilt_y), 0, np.sin(tilt_y)],
    [0, 1, 0],
    [-np.sin(tilt_y), 0, np.cos(tilt_y)]
  ])

  R = Ry @ Rx
  rotated = corners @ R.T
  projected = (rotated[:, :2] / (rotated[:, 2:3] / f + 1))
  projected = projected * scale + np.array([
    dst_width / 2 + offset_x,
    dst_height / 2 + offset_y
  ])
  projected = projected.astype(np.float32)
  return projected

def random_transform(img):
  """Applies a random perspective transform to the image and background.
  
  The background will be warped with the same transform and tiled so that
  it fills the entire output region.
  """

  # Ensure both images are 3-channel before any color operations
  if len(img.shape) == 2:  # grayscale → BGR
      img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
  
  # Grayscale
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = cv2.GaussianBlur(img, (5, 5), 0)
  img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


  h, w = img.shape[:2]
  dst_width, dst_height = w, h

  src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
  grid_src = np.float32([[w*0.1, h*0.1], [w*0.9, h*0.1], [w*0.9, h*0.9], [w*0.1, h*0.9]])

  # --- Random camera parameters ---
  max_tilt = np.pi / 6
  max_offset = 0.2
  min_scale = 0.2
  max_scale = 0.4

  scale = np.random.uniform(min_scale, max_scale)
  tilt_x = (np.random.rand() - 0.5) * 2 * max_tilt
  tilt_y = (np.random.rand() - 0.5) * 2 * max_tilt
  offset_x = (np.random.rand() - 0.5) * 2 * max_offset * dst_width
  offset_y = (np.random.rand() - 0.5) * 2 * max_offset * dst_height

  f = min(dst_width, dst_height) * 1.5
  cx, cy = w / 2, h / 2

  corners_3d = np.float32([
    [0, 0, 0],
    [w, 0, 0],
    [w, h, 0],
    [0, h, 0]
  ]) - np.float32([cx, cy, 0])

  gcorners_3d = np.float32([
    [w*0.06, h*0.15, 0],
    [w*0.92, h*0.15, 0],
    [w*0.92, h*0.95, 0],
    [w*0.06, h*0.95, 0]
  ]) - np.float32([cx, cy, 0])

  dst = apply_proj(corners_3d, f, tilt_x, tilt_y, offset_x, offset_y, dst_width, dst_height, scale)
  gdst = apply_proj(gcorners_3d, f, tilt_x, tilt_y, offset_x, offset_y, dst_width, dst_height, scale)

  M = cv2.getPerspectiveTransform(src, dst)

  # Randomly darken image
  darkness = np.random.uniform(0.5, 1.2)
  darkness_rgb = np.clip(np.array([209, 209, 209]) * darkness, 0, 255)
  img = np.clip(img * darkness, 0, 255)

  # --- Warp main image ---
  warped = cv2.warpPerspective(
    img, M, (dst_width, dst_height),
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=tuple(darkness_rgb.tolist())
  )

  # --- Warp background (if given) ---
  # if bg is not None:
  #   if bg_scale != 1.0:
  #     new_w = int(bg.shape[1] * bg_scale)
  #     new_h = int(bg.shape[0] * bg_scale)
  #     bg = cv2.resize(bg, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

  #   # Tile the background to ensure it's larger than needed
  #   # (at least 3×3 times the output size so we can offset without black)
  #   tiled_bg = np.tile(bg, (2, 2, 1))
  #   th, tw = tiled_bg.shape[:2]

  #   # Warp the big background the same way
  #   bg_warped = cv2.warpPerspective(
  #     tiled_bg, M, (dst_width, dst_height),
  #     flags=cv2.INTER_LINEAR,
  #     borderMode=cv2.BORDER_WRAP  # repeat when out of bounds
  #   )
  # else:
  #   bg_warped = np.zeros_like(warped)

  out = warped
  out = out * (1 - np.random.random((dst_height, dst_width, 1)) * 0.2)
  out = out.astype(np.float32)

  # Debug display grid points
  # cv2.line(out, (int(gdst[0][0]), int(gdst[0][1])), (int(gdst[1][0]), int(gdst[1][1])), (0, 0, 255), 2)
  # cv2.line(out, (int(gdst[1][0]), int(gdst[1][1])), (int(gdst[2][0]), int(gdst[2][1])), (0, 0, 255), 2)
  # cv2.line(out, (int(gdst[2][0]), int(gdst[2][1])), (int(gdst[3][0]), int(gdst[3][1])), (0, 0, 255), 2)
  # cv2.line(out, (int(gdst[3][0]), int(gdst[3][1])), (int(gdst[0][0]), int(gdst[0][1])), (0, 0, 255), 2)

  out = out[:, :, 1][:, :, np.newaxis]
  
  return out.astype(np.uint8), np.ravel(gdst)


