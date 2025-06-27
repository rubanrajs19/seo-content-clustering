try:
    import numpy
    import pandas
    import sklearn
    import matplotlib
    import umap
    import transformers
    import openpyxl

    print("✅ All required libraries are installed!")
except ImportError as e:
    print("❌ Missing a library:", e.name)
