which python3
python3 runner.py \
	--wandb False \
	--folder_name reward_evaluation_new \
	--num_simulation_steps 10 \
	--price_generation_function constant_peak_day_prices_generation_function \
	--offset_multiplier 0.01 \
	--off_peak_offset_multiplier 0.01 \
	--prosumer_noise_scale 0.1 \
	--generate_batch_data True \
	--generation_noise_scale 0.1
