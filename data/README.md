# Data Directory

This folder is populated by `src/create_sign_lang_data.py`.

After running the script, it will have the structure:

```
data/
├── 0/          # class 0  (e.g. letter "a")
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
├── 1/          # class 1  (e.g. letter "b")
│   └── ...
└── ...
```

> **Note:** Raw images are excluded from version control (see `.gitignore`).
> Run `src/create_sign_lang_data.py` to regenerate them.
