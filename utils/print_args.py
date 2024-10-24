import sys


def print_args(args):
    """
    Display the configuration arguments in a structured and formatted manner.

    Parameters:
        args: An object containing all configuration attributes.
    """

    def print_section(title: str, fields: list):
        """
        Print a section with a bold title and formatted key-value pairs.

        Parameters:
            title (str): The title of the section.
            fields (list): A list of lists, where each sublist contains tuples of key-value pairs.
        """
        # Print the section title in bold
        print(f"\033[1m{title}\033[0m")

        # Iterate through each line of key-value pairs
        for line in fields:
            try:
                # Convert all values to strings to avoid formatting issues
                formatted_line = "  " + "".join([f"{key:<20}{str(value):<20}" for key, value in line])
                print(formatted_line)
            except Exception as e:
                # Handle any unexpected formatting errors
                print(f"  Error formatting line {line}: {e}", file=sys.stderr)

        # Add an empty line for spacing
        print()

    # -------------------- Basic Configuration --------------------
    basic_fields = [
        [("Task Name:", args.task_name), ("Is Training:", args.is_training)],
        [("Model ID:", args.model_id), ("Model:", args.model)],
    ]
    print_section("Basic Config", basic_fields)

    # -------------------- Data Loader Configuration --------------------
    data_loader_fields = [
        [("Data:", args.data), ("Root Path:", args.root_path)],
        [("Data Path:", args.data_path), ("Features:", args.features)],
        [("Target:", args.target), ("Freq:", args.freq)],
        [("Checkpoints:", args.checkpoints)],
    ]
    print_section("Data Loader", data_loader_fields)

    # -------------------- Task-Specific Configurations --------------------
    if args.task_name in ['long_term_forecast', 'short_term_forecast']:
        forecasting_fields = [
            [("Seq Len:", args.seq_len), ("Label Len:", args.label_len)],
            [("Pred Len:", args.pred_len), ("Seasonal Patterns:", args.seasonal_patterns)],
            [("Inverse:", args.inverse)],
        ]
        print_section("Forecasting Task", forecasting_fields)

    # -------------------- Model Parameters --------------------
    model_params_fields = [
        [("Top k:", args.top_k), ("Num Kernels:", args.num_kernels)],
        [("Enc In:", args.enc_in), ("Dec In:", args.dec_in)],
        [("C Out:", args.c_out), ("d model:", args.d_model)],
        [("n heads:", args.n_heads), ("e layers:", args.e_layers)],
        [("d layers:", args.d_layers), ("d FF:", args.d_ff)],
        [("Moving Avg:", args.moving_avg), ("Factor:", args.factor)],
        [("Distil:", args.distil), ("Dropout:", args.dropout)],
        [("Embed:", args.embed), ("Activation:", args.activation)],
    ]
    print_section("Model Parameters", model_params_fields)

    # -------------------- Run Parameters --------------------
    run_params_fields = [
        [("Num Workers:", args.num_workers), ("Itr:", args.itr)],
        [("Train Epochs:", args.train_epochs), ("Batch Size:", args.batch_size)],
        [("Patience:", args.patience), ("Learning Rate:", args.learning_rate)],
        [("Des:", args.des), ("Loss:", args.loss)],
        [("Lradj:", args.lradj), ("Use Amp:", args.use_amp)],
    ]
    print_section("Run Parameters", run_params_fields)

    # -------------------- GPU Configuration --------------------
    gpu_fields = [
        [("Use GPU:", args.use_gpu), ("GPU:", args.gpu)],
        [("Use Multi GPU:", args.use_multi_gpu), ("Devices:", args.devices)],
    ]
    print_section("GPU", gpu_fields)

    # -------------------- De-stationary Projector Parameters --------------------
    p_hidden_dims_str = ', '.join(map(str, args.p_hidden_dims))
    destationary_fields = [
        [("P Hidden Dims:", p_hidden_dims_str), ("P Hidden Layers:", args.p_hidden_layers)],
    ]
    print_section("De-stationary Projector Params", destationary_fields)
