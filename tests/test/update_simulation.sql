UPDATE hardware_platforms SET is_simulated = TRUE, simulation_reason = 'Hardware not available' WHERE hardware_type = 'rocm';
