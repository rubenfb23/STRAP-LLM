from llm_instrumentation.core.config import InstrumentationConfig
from llm_instrumentation.core.framework import InstrumentationFramework
from llm_instrumentation.core.hooks import HookGranularity
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

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

# Prepare input
sample_text = "The future of artificial intelligence is"
input_ids = tokenizer.encode(sample_text, return_tensors="pt")

# Run inference with data capture
with framework.capture_activations("output.stream"):
    outputs = model.generate(input_ids, max_length=100)

# Analyze captured data
analysis = framework.analyze_activations("output.stream")
