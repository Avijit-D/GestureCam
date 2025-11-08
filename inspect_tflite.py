from tflite_support import flatbuffers
from tflite_support import metadata as _metadata

model_path = "your_model.tflite"

try:
    displayer = _metadata.MetadataDisplayer.with_model_file(model_path)
    print("✅ Model metadata found!\n")
    print(displayer.get_metadata_json())
except Exception as e:
    print("⚠️ No metadata or old TFLite format.\n", e)
