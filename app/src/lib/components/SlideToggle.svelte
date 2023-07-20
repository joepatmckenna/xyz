<script>
	let checked = false;
</script>

<label class="switch">
	<input type="checkbox" bind:checked />
	<span class="slider" />
</label>

<style>
	.switch {
		position: relative;
		display: inline-block;
		width: 48px;
		height: 24px;
	}

	.switch input {
		opacity: 0;
		width: 0;
		height: 0;
	}

	.slider {
		position: absolute;
		top: 0;
		left: 0;
		right: 0;
		bottom: 0;
		background-color: #ccc;
		-webkit-transition: 0.4s;
		transition: 0.4s;
		border-radius: 28px;
		box-shadow: 0 0 1px #000;
	}

	.slider:before {
		position: absolute;
		content: '';
		height: 16px;
		width: 16px;
		left: 4px;
		bottom: 4px;
		background-color: #fff;
		-webkit-transition: 0.4s;
		transition: 0.4s;
		border-radius: 50%;
	}

	input:not(:checked) + .slider {
		background-color: #ccc;
	}

	input:checked + .slider {
		background-color: #fff;
	}

	input:checked + .slider:before {
		background-color: #ccc;
	}

	input:checked + .slider:before {
		-webkit-transform: translateX(24px);
		-ms-transform: translateX(24px);
		transform: translateX(24px);
	}
</style>
