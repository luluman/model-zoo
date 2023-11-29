from PIL import Image
from pathlib import Path


roots = ['DIV2K_test_LR_bicubic']

for root in roots:
    root = Path(root)
    save_dir = root / 'X8'
    save_dir.mkdir(exist_ok=True)

    for path in root.glob('X4/*.png'):
        im = Image.open(path)
        size_new = [s//2 for s in im.size]
        name_new = path.name.replace('x4', 'x8')
        im.resize(size_new, resample=Image.BICUBIC).save(save_dir/name_new)
