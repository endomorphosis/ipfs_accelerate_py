# Comprehensive Model Compatibility Matrix

**Generated:** {{ generated_date }} | **Models:** {{ total_models }} | **Hardware Platforms:** {{ total_hardware_platforms }}

## Compatibility Levels
- ✅ Full Support: Fully tested and optimized
- ⚠️ Partial Support: Works with some limitations
- 🔶 Limited Support: Basic functionality only
- ❌ Not Supported: Known incompatibility

## Matrix by Model Family

{% for modality, models in models_by_modality.items() %}
### {{ modality|title }} Models

| Model | Family | Parameters | {% for hw in hardware_platforms %}{{ hw }} | {% endfor %}Notes |
|-------|--------|------------|{% for hw in hardware_platforms %}------|{% endfor %}-------|
{% for model in models %}| {{ model.model_name }} | {{ model.model_family }} | {{ model.parameters_million }}M | {% for hw in hardware_platforms %}{{ model.get(hw, '❌') }} | {% endfor %}{{ model.compatibility_notes }} |
{% endfor %}

{% endfor %}

## Performance Summary

{% if include_performance %}
### Average Throughput (items/sec) by Hardware Platform

| Model Family | {% for hw in hardware_platforms %}{{ hw }} | {% endfor %}
|-------------|{% for hw in hardware_platforms %}------|{% endfor %}
{% for family, metrics in performance_by_family.items() %}| {{ family }} | {% for hw in hardware_platforms %}{{ metrics.get(hw, {}).get('avg_throughput', 'N/A') }} | {% endfor %}
{% endfor %}

### Average Latency (ms) by Hardware Platform

| Model Family | {% for hw in hardware_platforms %}{{ hw }} | {% endfor %}
|-------------|{% for hw in hardware_platforms %}------|{% endfor %}
{% for family, metrics in performance_by_family.items() %}| {{ family }} | {% for hw in hardware_platforms %}{{ metrics.get(hw, {}).get('avg_latency', 'N/A') }} | {% endfor %}
{% endfor %}

### Memory Usage (MB) by Hardware Platform

| Model Family | {% for hw in hardware_platforms %}{{ hw }} | {% endfor %}
|-------------|{% for hw in hardware_platforms %}------|{% endfor %}
{% for family, metrics in performance_by_family.items() %}| {{ family }} | {% for hw in hardware_platforms %}{{ metrics.get(hw, {}).get('avg_memory', 'N/A') }} | {% endfor %}
{% endfor %}
{% endif %}

## Recommendations by Model Type

{% for modality, recs in recommendations.items() %}
### {{ modality|title }} Models

{{ recs.summary }}

**Best Hardware Platform**: {{ recs.best_platform }}

**Recommended Configurations**:
{% for config in recs.configurations %}
- {{ config }}
{% endfor %}

{% endfor %}

---

This compatibility matrix is automatically generated from the benchmark database. For detailed implementation guides and optimization recommendations, please refer to the [Hardware Selection Guide](HARDWARE_SELECTION_GUIDE.md) and [Hardware Benchmarking Guide](HARDWARE_BENCHMARKING_GUIDE.md).