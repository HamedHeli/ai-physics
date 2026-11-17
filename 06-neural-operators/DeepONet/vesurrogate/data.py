"""Load Abaqus ramp-and-hold result files.

Expected file name: Abaqus_Results_<size>_<depth>mm.txt, e.g. Abaqus_Results_495_1mm.txt
  <size>  indenter diameter phi in mm. Digits only means the decimal point goes after
          the first digit (495 -> 4.95, 633 -> 6.33, 760 -> 7.60, 888 -> 8.88).
          "4.95" or "4p95" are also accepted.
  <depth> final indentation depth delta in mm ("1mm", "2.5mm", "2p5mm").
Optional suffix after the depth (e.g. _t012 for ramp time 0.12 s, or a material tag)
is kept in `tag`.

File content: two whitespace-separated columns, time [s] and the indenter output.
In the sample file the second column is negative and peaks at the end of the ramp,
then relaxes: it is the indenter normal stress (paper: "average stress of elements
beneath the indenter", most likely in MPa) or reaction force. The sign is flipped so
the curve is positive. Abaqus writes the step boundary twice (t = tau0 appears twice),
which is how the ramp time tau0 is detected.
"""
import glob
import os
import re

import numpy as np

_NAME = re.compile(r"Abaqus_Results_([0-9.p]+)_([0-9.p]+)mm(.*)\.txt$", re.I)


def _num(s, size=False):
    s = s.replace("p", ".")
    if "." in s:
        return float(s)
    if size and len(s) > 1:
        return float(s[0] + "." + s[1:])
    return float(s)


def parse_name(path):
    m = _NAME.search(os.path.basename(path))
    if not m:
        raise ValueError(f"cannot parse indenter size/depth from {path}")
    return _num(m.group(1), size=True), _num(m.group(2)), m.group(3).strip("_")


def load_curve(path):
    phi, delta, tag = parse_name(path)
    raw = np.loadtxt(path)
    t, y = raw[:, 0], raw[:, 1]
    # ramp end = duplicated time stamp (step boundary); fall back to |peak|
    dup = np.where(np.diff(t) == 0)[0]
    tau0 = float(t[dup[0]]) if len(dup) else float(t[np.argmax(np.abs(y))])
    t, idx = np.unique(t, return_index=True)
    y = y[idx]
    if y[np.argmax(np.abs(y))] < 0:
        y = -y
    return dict(path=path, phi=phi, delta=delta, d=delta / phi, tau0=tau0, tag=tag, t=t, y=y)


def load_folder(folder):
    files = sorted(glob.glob(os.path.join(folder, "Abaqus_Results_*mm*.txt")))
    return [load_curve(f) for f in files]
