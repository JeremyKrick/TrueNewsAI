from transformers import pipeline
from sagemaker_inference import content_types, decoder, encoder, default_inference_handler
import json

# Initialize the model
model = pipeline('sentiment-analysis')

class HuggingFaceHandler(default_inference_handler.DefaultInferenceHandler):
    def default_model_fn(self, model_dir):
        return model

    def default_input_fn(self, input_data, content_type):
        if content_type == content_types.JSON:
            return json.loads(input_data)
        else:
            raise ValueError(f"Unsupported content type: {content_type}")

    def default_predict_fn(self, data, model):
        return model(data)

    def default_output_fn(self, prediction, accept):
        if accept == content_types.JSON:
            return encoder.encode(prediction, accept)
        else:
            raise ValueError(f"Unsupported accept type: {accept}")

# Define the entry point for the model server
if __name__ == "__main__":
    import sagemaker_inference
    sagemaker_inference.default_handler_service.HANDLER = HuggingFaceHandler()
    serve()
