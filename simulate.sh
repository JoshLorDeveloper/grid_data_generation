which python3
python3 runner.py \
	--wandb True \
	--folder_name reward_evaluation \
	--num_simulation_steps 100 \
	--price_generation_function constant_peak_prices_generation_function \
	--offset_multiplier 0.0001 \
	# --prosumer_noise_scale 0.1 \
	# --generation_noise_scale 0.05 \
	# --generate_batch_data \
