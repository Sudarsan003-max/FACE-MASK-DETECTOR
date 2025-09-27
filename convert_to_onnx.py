import tensorflow as tf
import tf2onnx
import os

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def load_custom_model(model_path):
    """Load a Keras model with custom objects and fix compatibility issues."""
    # Define custom objects with compatibility fixes
    custom_objects = {
        'GlorotUniform': tf.keras.initializers.GlorotUniform(),
        'relu6': tf.keras.layers.ReLU(6.0),
        'hard_swish': tf.keras.layers.ReLU(6.0),  # Approximation
    }
    
    # First try loading with custom objects (Keras 2.x style)
    try:
        # Try with Keras 2.x style loading first
        return tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
    except Exception as e:
        print(f"Failed to load model with Keras 2.x loader: {e}")
    
    # Try with Keras 3.x style loading
    try:
        # Create a custom GlorotUniform that ignores dtype
        class FixedGlorotUniform(tf.keras.initializers.GlorotUniform):
            def __init__(self, **kwargs):
                if 'dtype' in kwargs:
                    del kwargs['dtype']
                super().__init__(**kwargs)
        
        custom_objects['GlorotUniform'] = FixedGlorotUniform()
        
        # Try loading with the fixed initializer
        return tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
    except Exception as e:
        print(f"Failed to load model with Keras 3.x loader: {e}")
    
    # If that fails, try loading the model architecture and weights separately
    try:
        import json
        import h5py
        
        # Load the model config
        with h5py.File(model_path, 'r') as f:
            model_config = f.attrs['model_config']
            if hasattr(model_config, 'decode'):
                model_config = model_config.decode('utf-8')
            model_config = json.loads(model_config)
        
        # Fix the GlorotUniform initializer in the config
        def fix_initializer(config):
            if isinstance(config, dict):
                if 'config' in config and 'class_name' in config:
                    if config['class_name'] == 'GlorotUniform':
                        if 'dtype' in config['config']:
                            del config['config']['dtype']
                for k, v in config.items():
                    config[k] = fix_initializer(v)
            elif isinstance(config, (list, tuple)):
                return [fix_initializer(x) for x in config]
            return config
        
        fixed_config = fix_initializer(model_config)
        
        # Rebuild the model using the public API
        model = tf.keras.Model.from_config(fixed_config, custom_objects=custom_objects)
        
        # Load weights
        model.load_weights(model_path, by_name=True, skip_mismatch=True)
        return model
    except Exception as e:
        print(f"Failed to load model with manual fix: {e}")
        raise

def convert_keras_to_onnx(h5_model_path, output_onnx_path):
    """
    Convert a Keras .h5 model to ONNX format.
    
    Args:
        h5_model_path (str): Path to the input .h5 Keras model
        output_onnx_path (str): Path where to save the ONNX model
    """
    print(f"Loading Keras model from {h5_model_path}...")
    
    # Load the Keras model
    try:
        model = load_custom_model(h5_model_path)
        
        # Print model summary
        print("\nModel Summary:")
        print(model.summary())
        
        # Get input shape from the model
        input_shape = model.input_shape[1:]
        print(f"\nInput shape: {input_shape}")
        
        # Convert the model to ONNX
        print("\nConverting to ONNX format...")
        model_proto, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=[
                tf.TensorSpec(shape=[None] + list(input_shape), dtype=tf.float32, name='input')
            ],
            output_path=output_onnx_path,
            opset=13
        )
        
        print(f"\nSuccessfully converted model to ONNX format: {output_onnx_path}")
        print(f"Input name: {model_proto.graph.input[0].name}")
        print(f"Output name: {model_proto.graph.output[0].name}")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Path to your Keras model
    h5_model_path = "mask_detector.h5"
    
    # Output ONNX model path
    onnx_output_path = "mask_detector.onnx"
    
    # Convert the model
    convert_keras_to_onnx(h5_model_path, onnx_output_path)
