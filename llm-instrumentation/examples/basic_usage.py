from llm_instrumentation.core.hooks import HookGranularity
from llm_instrumentation import InstrumentationFramework
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure instrumentation
config = InstrumentationConfig(
    granularity=HookGranularity.ATTENTION_ONLY,
    compression_algorithm="lz4",
    target_throughput_gbps=2.0,
    max_memory_gb=24,
)

# Initialize framework
framework = InstrumentationFramework(config)

# Instrument model
framework.instrument_model(model)

# Run inference with data capture
with framework.capture_activations("output.stream"):
    outputs = model.generate(input_ids, max_length=100)

# Analyze captured data
analysis = framework.analyze_activations("output.stream")
