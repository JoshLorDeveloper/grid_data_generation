which python3
python3 runner.py \
	--wandb True \
	--folder_name reward_evaluation \
	--pv_size 20 \
	--num_simulation_steps 365 \
	--price_generation_function constant_peak_day_prices_generation_function \
	--offset_multiplier 0.1 \
	--off_peak_offset_multiplier 0.01
	# --prosumer_noise_scale 0.1 \
	# --generation_noise_scale 0.05 \
	# --generate_batch_data \
