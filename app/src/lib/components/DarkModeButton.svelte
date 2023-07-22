<script>
	import { onMount } from 'svelte';
	import { MoonRegular, SunRegular } from 'svelte-awesome-icons';
	import { darkMode } from '$lib/utils';

	onMount(async () => {
		const response = await fetch('/api/theme');
		darkMode.set(await response.json());
	});

	const update = (dark) => {
		dark = !dark;
		fetch('/api/theme', {
			method: 'POST',
			body: JSON.stringify({ darkMode: dark })
		});
		return dark;
	};

	const toggle = () => darkMode.update(update);
</script>

<!-- svelte-ignore a11y-click-events-have-key-events -->
<!-- svelte-ignore a11y-no-static-element-interactions -->
<div on:click={toggle}>
	{#if $darkMode}
		<SunRegular />
	{:else}
		<MoonRegular />
	{/if}
</div>
