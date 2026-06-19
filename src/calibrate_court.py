"""
Court calibration for padel — homography (pixels → real-world metres).

Padel court outer boundary: 20.00 m long × 10.00 m wide.
Pick the 4 OUTER corners in order: TL → TR → BR → BL.

Usage:
    python src/calibrate_court.py --source match.mp4 --frame 0 \
        --corners "x1,y1 x2,y2 x3,y3 x4,y4" --out court.npz
"""
import argparse
import cv2
import numpy as np

COURT_W_M = 10.00   # width (across the net)
COURT_L_M = 20.00   # length (end to end)

COURT_PTS_M = np.float32([
    [0.0,       0.0      ],   # TL
    [COURT_W_M, 0.0      ],   # TR
    [COURT_W_M, COURT_L_M],   # BR
    [0.0,       COURT_L_M],   # BL
])

PREVIEW_PPM = 25   # pixels-per-metre for top-down preview image


class CourtMapper:
    """Maps image pixels to court metres and estimates real-world speed."""

    def __init__(self, H, image_corners, frame_size):
        self.H = np.asarray(H, dtype=np.float64)
        self.image_corners = np.asarray(image_corners, dtype=np.float32)
        self.frame_size = tuple(frame_size)

    def to_metres(self, pts):
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def contains(self, pts, margin_m: float = 0.0) -> np.ndarray:
        m = self.to_metres(np.asarray(pts, dtype=np.float32))
        return (
            (m[:, 0] >= -margin_m) & (m[:, 0] <= COURT_W_M + margin_m) &
            (m[:, 1] >= -margin_m) & (m[:, 1] <= COURT_L_M + margin_m)
        )

    def speed_kmh(self, a, b, dt: float) -> float:
        am = self.to_metres([a])[0]
        bm = self.to_metres([b])[0]
        dist = float(np.linalg.norm(bm - am))
        return (dist / dt) * 3.6 if dt > 0 else 0.0

    def save(self, path: str):
        np.savez(path, H=self.H,
                 image_corners=self.image_corners,
                 frame_size=np.array(self.frame_size))

    @classmethod
    def load(cls, path: str) -> 'CourtMapper':
        d = np.load(path)
        return cls(d['H'], d['image_corners'], tuple(d['frame_size'].tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--frame', type=int, default=0)
    ap.add_argument('--corners', help='"x1,y1 x2,y2 x3,y3 x4,y4"')
    ap.add_argument('--out', default='court.npz')
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError('Could not read frame')

    h, w = frame.shape[:2]
    if args.corners:
        pts = [[float(v) for v in p.split(',')] for p in args.corners.split()]
        corners = np.float32(pts)
    else:
        print('Click TL, TR, BR, BL — press any key after each')
        corners = []
        def click(e, x, y, *_):
            if e == cv2.EVENT_LBUTTONDOWN:
                corners.append([x, y])
                cv2.circle(frame, (x,y), 6, (0,255,0), -1)
                cv2.imshow('court', frame)
        cv2.imshow('court', frame); cv2.setMouseCallback('court', click)
        while len(corners) < 4:
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        corners = np.float32(corners)

    H, _ = cv2.findHomography(corners, COURT_PTS_M)
    mapper = CourtMapper(H, corners, (w, h))
    mapper.save(args.out)

    # top-down preview
    pw = int(COURT_W_M * PREVIEW_PPM)
    ph = int(COURT_L_M * PREVIEW_PPM)
    H_inv = np.linalg.inv(H)
    topdown = cv2.warpPerspective(frame, H_inv * PREVIEW_PPM, (pw, ph))
    cv2.imwrite('court_topdown.jpg', topdown)
    print(f'Saved {args.out} and court_topdown.jpg')


if __name__ == '__main__':
    main()
